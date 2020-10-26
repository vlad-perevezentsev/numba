#include "pipelines/lower_to_llvm.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/Triple.h>
#include <llvm/Support/Host.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

#include "plier/dialect.hpp"

#include "base_pipeline.hpp"
#include "pipeline_registry.hpp"

#include "utils.hpp"

namespace
{
const mlir::LowerToLLVMOptions &getLLVMOptions()
{
    static mlir::LowerToLLVMOptions options = []()
    {
        llvm::InitializeNativeTarget();
        auto triple = llvm::sys::getProcessTriple();
        std::string err_str;
        auto target = llvm::TargetRegistry::lookupTarget(triple, err_str);
        if (nullptr == target)
        {
            report_error(llvm::Twine("Unable to get target: ") + err_str);
        }
        llvm::TargetOptions target_opts;
        std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(triple, llvm::sys::getHostCPUName(), "", target_opts, llvm::None));
        mlir::LowerToLLVMOptions opts;
        opts.dataLayout = machine->createDataLayout();
        opts.useBarePtrCallConv = true;
        return opts;
    }();
    return options;
}

struct LLVMTypeHelper
{
    LLVMTypeHelper(mlir::MLIRContext& ctx):
        type_converter(&ctx) {}

    mlir::LLVM::LLVMType i(unsigned bits)
    {
        return mlir::LLVM::LLVMIntegerType::get(&type_converter.getContext(), bits);
    }

    mlir::LLVM::LLVMType ptr(mlir::Type type)
    {
        assert(static_cast<bool>(type));
        auto ll_type = type_converter.convertType(type).cast<mlir::LLVM::LLVMType>();
        assert(static_cast<bool>(ll_type));
        return mlir::LLVM::LLVMPointerType::get(ll_type);
    }

    mlir::MLIRContext& get_context()
    {
        return type_converter.getContext();
    }

    mlir::LLVMTypeConverter& get_type_converter()
    {
        return type_converter;
    }

private:
    mlir::LLVMTypeConverter type_converter;
};

mlir::Type getExceptInfoType(LLVMTypeHelper& type_helper)
{
    mlir::LLVM::LLVMType elems[] = {
        type_helper.ptr(type_helper.i(8)),
        type_helper.i(32),
        type_helper.ptr(type_helper.i(8)),
    };
    return mlir::LLVM::LLVMStructType::getLiteral(&type_helper.get_context(), elems);
}

void fix_func_sig(LLVMTypeHelper& type_helper, mlir::FuncOp func)
{
    auto old_type = func.getType();
    assert(old_type.getNumResults() == 1);
    auto& ctx = *old_type.getContext();
    llvm::SmallVector<mlir::Type, 8> args;

    auto ptr = [&](auto arg)
    {
        return type_helper.ptr(arg);
    };

    unsigned index = 0;
    auto add_arg = [&](mlir::Type type)
    {
        args.push_back(type);
        func.getBody().insertArgument(index, type);
        ++index;
    };

    add_arg(ptr(old_type.getResult(0)));
    add_arg(ptr(ptr(getExceptInfoType(type_helper))));

    auto old_args = old_type.getInputs();
    std::copy(old_args.begin(), old_args.end(), std::back_inserter(args));
    auto ret_type = mlir::IntegerType::get(32, &ctx);
    func.setType(mlir::FunctionType::get(args, ret_type, &ctx));
}

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::ReturnOp>
{
    ReturnOpLowering(mlir::MLIRContext* ctx, mlir::TypeConverter& converter):
        OpRewritePattern(ctx), type_converter(converter) {}

    mlir::LogicalResult matchAndRewrite(mlir::ReturnOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        auto insert_ret = [&]()
        {
            auto ctx = op.getContext();
            auto ret_type = mlir::IntegerType::get(32, ctx);
            auto ll_ret_type = mlir::LLVM::LLVMIntegerType::get(ctx, 32);
            mlir::Value ret = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), ll_ret_type, mlir::IntegerAttr::get(ret_type, 0));
            rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, ret);
        };

        if (op.getNumOperands() == 0)
        {
            rewriter.setInsertionPoint(op);
            insert_ret();
            return mlir::success();
        }
        else if (op.getNumOperands() == 1)
        {
            rewriter.setInsertionPoint(op);
            auto addr = op.getParentRegion()->front().getArgument(0);
            auto val = op.getOperand(0);
            auto ll_ret_type = type_converter.convertType(val.getType());
            assert(static_cast<bool>(ll_ret_type));
            auto ll_val = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), ll_ret_type, val); // TODO: hack to make verifier happy
            rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), ll_val, addr);
            insert_ret();
            return mlir::success();
        }
        else
        {
            return mlir::failure();
        }
    }

private:
    mlir::TypeConverter& type_converter;
};

struct RemoveBitcasts : public mlir::OpRewritePattern<mlir::LLVM::BitcastOp>
{
    using mlir::OpRewritePattern<mlir::LLVM::BitcastOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::LLVM::BitcastOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        if (op.getType() == op.getOperand().getType())
        {
            rewriter.replaceOp(op, op.getOperand());
            return mlir::success();
        }
        return mlir::failure();
    }
};

class CheckForPlierTypes :
    public mlir::PassWrapper<CheckForPlierTypes, mlir::OperationPass<void>>
{
    void runOnOperation() override
    {
        markAllAnalysesPreserved();
        getOperation()->walk([&](mlir::Operation* op)
        {
            if (op->getName().getDialect() == plier::PlierDialect::getDialectNamespace())
            {
                op->emitOpError(": not all plier ops were translated\n");
                signalPassFailure();
                return;
            }

            auto check_type = [](mlir::Type type)
            {
                return type.isa<plier::PyType>();
            };

            if (llvm::any_of(op->getResultTypes(), check_type) ||
                llvm::any_of(op->getOperandTypes(), check_type))
            {
                op->emitOpError(": not all plier types were translated\n");
                signalPassFailure();
            }
        });
    }
};

class LLVMFunctionPass : public mlir::OperationPass<mlir::LLVM::LLVMFuncOp>
{
public:
  using OperationPass<mlir::LLVM::LLVMFuncOp>::OperationPass;

  /// The polymorphic API that runs the pass over the currently held function.
  virtual void runOnFunction() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final {
    if (!getFunction().isExternal())
      runOnFunction();
  }

  /// Return the current function being transformed.
  mlir::LLVM::LLVMFuncOp getFunction() { return this->getOperation(); }
};

struct PreLLVMLowering : public mlir::PassWrapper<PreLLVMLowering, mlir::FunctionPass>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnFunction() override final
    {
        LLVMTypeHelper type_helper(getContext());

        mlir::OwningRewritePatternList patterns;
        auto apply_conv = [&]()
        {
            (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
        };

        auto func = getFunction();
        fix_func_sig(type_helper, func);

        patterns.insert<ReturnOpLowering>(&getContext(),
                                          type_helper.get_type_converter());

        apply_conv();
    }
};

struct PostLLVMLowering :
    public mlir::PassWrapper<PostLLVMLowering, LLVMFunctionPass>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnFunction() override final
    {
        mlir::OwningRewritePatternList patterns;
        auto apply_conv = [&]()
        {
            (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
        };

        // Remove redundant bitcasts we have created on PreLowering
        patterns.insert<RemoveBitcasts>(&getContext());

        apply_conv();
    }
};

void populate_lower_to_llvm_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<CheckForPlierTypes>());
    pm.addPass(std::make_unique<PreLLVMLowering>());
    pm.addPass(mlir::createLowerToLLVMPass(getLLVMOptions()));
    pm.addPass(std::make_unique<PostLLVMLowering>());
}
}


void register_lower_to_llvm_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_lower_lowering_stage();
        sink(lower_to_llvm_pipeline_name(), {stage.begin}, {stage.end}, &populate_lower_to_llvm_pipeline);
    });
}

llvm::StringRef lower_to_llvm_pipeline_name()
{
    return "lower_to_llvm";
}
