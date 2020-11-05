#include "pipelines/lower_to_llvm.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Builders.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/DenseMap.h>
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

mlir::LLVM::LLVMStructType get_array_type(mlir::TypeConverter& converter, mlir::MemRefType type)
{
    assert(type);
    auto ctx = type.getContext();
    auto i8p = mlir::LLVM::LLVMType::getInt8Ty(ctx).getPointerTo();
    auto i64 = mlir::LLVM::LLVMType::getIntNTy(ctx, 64);
    auto data_type = converter.convertType(type.getElementType()).cast<mlir::LLVM::LLVMType>();
    assert(data_type);
    auto shape_type = mlir::LLVM::LLVMArrayType::get(i64, static_cast<unsigned>(type.getRank()));
    const mlir::LLVM::LLVMType members[] = {
        i8p, // 0, meminfo
        i8p, // 1, parent
        i64, // 2, nitems
        i64, // 3, itemsize
        data_type.getPointerTo(), // 4, data
        shape_type, // 5, shape
        shape_type, // 6, strides
    };
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
}

template<typename F>
void flatten_type(mlir::LLVM::LLVMType type, F&& func)
{
    if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>())
    {
        for (auto elem : struct_type.getBody())
        {
            flatten_type(elem, std::forward<F>(func));
        }
    }
    else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>())
    {
        auto elem = arr_type.getElementType();
        auto size = arr_type.getNumElements();
        for (unsigned i = 0 ; i < size; ++i)
        {
            flatten_type(elem, std::forward<F>(func));
        }
    }
    else
    {
        func(type);
    }
}

template<typename F>
mlir::Value unflatten(mlir::LLVM::LLVMType type, mlir::Location loc, mlir::OpBuilder& builder, F&& next_func)
{
    namespace mllvm = mlir::LLVM;
    if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>())
    {
        mlir::Value val = builder.create<mllvm::UndefOp>(loc, struct_type);
        for (auto elem : llvm::enumerate(struct_type.getBody()))
        {
            auto elem_index = builder.getI64ArrayAttr(static_cast<int64_t>(elem.index()));
            auto elem_type = elem.value();
            auto elem_val = unflatten(elem_type, loc, builder, std::forward<F>(next_func));
            val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val, elem_index);
        }
        return val;
    }
    else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>())
    {
        auto elem_type = arr_type.getElementType();
        auto size = arr_type.getNumElements();
        mlir::Value val = builder.create<mllvm::UndefOp>(loc, arr_type);
        for (unsigned i = 0 ; i < size; ++i)
        {
            auto elem_index = builder.getI64ArrayAttr(static_cast<int64_t>(i));
            auto elem_val = unflatten(elem_type, loc, builder, std::forward<F>(next_func));
            val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val, elem_index);
        }
        return val;
    }
    else
    {
        return next_func();
    }
}

std::string gen_conversion_func_name(mlir::MemRefType memref_type)
{
    assert(memref_type);
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << "__convert_memref_";
    memref_type.getElementType().print(ss);
    ss.flush();
    return ret;
}

const constexpr llvm::StringRef linkage_attr = "numba_linkage";

struct MemRefConversionCache
{
    mlir::FuncOp get_conversion_func(
        mlir::ModuleOp module, mlir::OpBuilder& builder, mlir::MemRefType memref_type,
        mlir::LLVM::LLVMStructType src_type, mlir::LLVM::LLVMStructType dst_type)
    {
        assert(memref_type);
        assert(src_type);
        assert(dst_type);
        auto it = cache.find(memref_type);
        if (it != cache.end())
        {
            auto func = it->second;
            assert(func.getType().getNumResults() == 1);
            assert(func.getType().getResult(0) == dst_type);
            return func;
        }
        auto func_name = gen_conversion_func_name(memref_type);
        auto func_type = mlir::FunctionType::get(src_type, dst_type, builder.getContext());
        auto loc = builder.getUnknownLoc();
        auto new_func = mlir::FuncOp::create(loc, func_name, func_type);
        new_func.setAttr(linkage_attr, mlir::StringAttr::get("internal", builder.getContext()));
        module.push_back(new_func);
        cache.insert({memref_type, new_func});
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto block = new_func.addEntryBlock();
        builder.setInsertionPointToStart(block);
        namespace mllvm = mlir::LLVM;
        mlir::Value arg = block->getArgument(0);
        auto extract = [&](unsigned index)
        {
            auto res_type = src_type.getBody()[index];
            auto i = builder.getI64ArrayAttr(index);
            return builder.create<mllvm::ExtractValueOp>(loc, res_type, arg, i);
        };
        auto ptr = extract(4);
        auto shape = extract(5);
        auto strides = extract(6);
        auto i64 = mllvm::LLVMIntegerType::get(builder.getContext(), 64);
        auto offset = builder.create<mllvm::ConstantOp>(loc, i64, builder.getI64IntegerAttr(0));
        mlir::Value res = builder.create<mllvm::UndefOp>(loc, dst_type);
        auto insert = [&](unsigned index, mlir::Value val)
        {
            auto i = builder.getI64ArrayAttr(index);
            res = builder.create<mllvm::InsertValueOp>(loc, res, val, i);
        };
        insert(0, ptr);
        insert(1, ptr);
        insert(2, offset);
        insert(3, shape);
        insert(4, strides);
        builder.create<mllvm::ReturnOp>(loc, res);
        return new_func;
    }
private:
    llvm::DenseMap<mlir::Type, mlir::FuncOp> cache;
};

llvm::StringRef get_linkage(mlir::Operation* op)
{
    assert(nullptr != op);
    if (auto attr = op->getAttr(linkage_attr).dyn_cast_or_null<mlir::StringAttr>())
    {
        return attr.getValue();
    }
    return {};
}

void fix_func_sig(LLVMTypeHelper& type_helper, mlir::FuncOp func)
{
    if (get_linkage(func) == "internal")
    {
        return;
    }
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
        auto ret = func.getBody().insertArgument(index, type);
        ++index;
        return ret;
    };

    MemRefConversionCache conversion_cache;

    mlir::OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&func.getBody().front());

    auto loc = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value, 8> new_args;
    auto process_arg = [&](mlir::Type type)
    {
        if (auto memref_type = type.dyn_cast<mlir::MemRefType>())
        {
            new_args.clear();
            auto arr_type = get_array_type(type_helper.get_type_converter(), memref_type);
            flatten_type(arr_type, [&](mlir::Type new_type)
            {
                new_args.push_back(add_arg(new_type));
            });
            auto it = new_args.begin();
            mlir::Value desc = unflatten(arr_type, loc, builder, [&]()
            {
                auto ret = *it;
                ++it;
                return ret;
            });

            auto mod = mlir::cast<mlir::ModuleOp>(func.getParentOp());
            auto dst_type = type_helper.get_type_converter().convertType(memref_type);
            assert(dst_type);
            auto conv_func = conversion_cache.get_conversion_func(mod, builder, memref_type, arr_type, dst_type.cast<mlir::LLVM::LLVMStructType>());
            auto converted = builder.create<mlir::CallOp>(loc, conv_func, desc).getResult(0);
            auto casted = builder.create<plier::CastOp>(loc, memref_type, converted);
            func.getBody().getArgument(index).replaceAllUsesWith(casted);
            func.getBody().eraseArgument(index);
        }
        else
        {
            args.push_back(type);
            ++index;
        }
    };

    add_arg(ptr(old_type.getResult(0)));
    add_arg(ptr(ptr(getExceptInfoType(type_helper))));

    auto old_args = old_type.getInputs();
//    std::copy(old_args.begin(), old_args.end(), std::back_inserter(args));
    for (auto arg : old_args)
    {
        process_arg(arg);
    }
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
        auto func = getFunction();
        fix_func_sig(type_helper, func);

        patterns.insert<ReturnOpLowering>(&getContext(),
                                          type_helper.get_type_converter());

        (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
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

        // Remove redundant bitcasts we have created on PreLowering
        patterns.insert<RemoveBitcasts>(&getContext());

        (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
};

struct LowerCasts : public mlir::OpConversionPattern<plier::CastOp>
{
    using mlir::OpConversionPattern<plier::CastOp>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(plier::CastOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 1);
        auto converter = getTypeConverter();
        assert(nullptr != converter);
        auto src_type = operands[0].getType();
        auto dst_type = converter->convertType(op.getType());
        if (src_type == dst_type)
        {
            rewriter.replaceOp(op, operands[0]);
            return mlir::success();
        }
        return mlir::failure();
    }
};

// Copypasted from mlir
struct LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
  LLVMLoweringPass(const mlir::LowerToLLVMOptions& opts):
    options(opts) {}

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    using namespace mlir;
    if (options.useBarePtrCallConv && options.emitCWrappers) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            options.dataLayout.getStringRepresentation(), [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();

    LLVMTypeConverter typeConverter(&getContext(), options);

    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    patterns.insert<LowerCasts>(typeConverter, &getContext());

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    m.setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
              StringAttr::get(options.dataLayout.getStringRepresentation(), m.getContext()));
  }

private:
  mlir::LowerToLLVMOptions options;
};

void populate_lower_to_llvm_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createLowerToCFGPass());
    pm.addPass(std::make_unique<CheckForPlierTypes>());
    pm.addNestedPass<mlir::FuncOp>(std::make_unique<PreLLVMLowering>());
    pm.addPass(std::make_unique<LLVMLoweringPass>(getLLVMOptions()));
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(std::make_unique<PostLLVMLowering>());
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
