#include "pipelines/plier_to_linalg.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include "plier/dialect.hpp"

#include "pipelines/plier_to_std.hpp"
#include "rewrites/type_conversion.hpp"

#include "base_pipeline.hpp"
#include "pipeline_registry.hpp"

#include <cctype>

namespace
{
bool parse_layout(llvm::StringRef& name)
{
    return name.consume_back("C"); // TODO
}

template<typename T>
bool consume_int_back(llvm::StringRef& name, T& result)
{
    unsigned len = 0;
    auto tmp_name = name;
    while (!tmp_name.empty() && std::isdigit(tmp_name.back()))
    {
        ++len;
        tmp_name = tmp_name.drop_back();
    }
    tmp_name = name.substr(name.size() - len);
    if (!tmp_name.consumeInteger<T>(10, result))
    {
        name = name.substr(0, name.size() - len);
        return true;
    }
    return false;
}

mlir::Type map_array_type(mlir::MLIRContext& ctx, mlir::TypeConverter& conveter,
                          llvm::StringRef& name)
{
    unsigned num_dims = 0;
    if (name.consume_front("array(") &&
        name.consume_back(")") &&
        parse_layout(name) &&
        name.consume_back(", ") &&
        name.consume_back("d") &&
        consume_int_back(name, num_dims) &&
        name.consume_back(", ") &&
        !name.empty())
    {
        if (auto type = conveter.convertType(plier::PyType::get(&ctx, name)))
        {
            llvm::SmallVector<int64_t, 8> shape(num_dims, -1);
            return mlir::MemRefType::get(shape, type);
        }
    }
    return nullptr;
}


mlir::Type map_plier_type(mlir::TypeConverter& converter, mlir::Type type)
{
    if (type.isa<plier::PyType>())
    {
        auto name = type.cast<plier::PyType>().getName();
        return map_array_type(*type.getContext(), converter, name);
    }
    return nullptr;
}

llvm::StringRef extract_bound_func_name(llvm::StringRef name)
{
    auto len = name.find(' ');
    return name.substr(0, len);
}

struct CallOpLowering : public mlir::OpRewritePattern<plier::PyCallOp>
{
    using check_t = mlir::LogicalResult(*)(llvm::StringRef, llvm::ArrayRef<mlir::Type>);
    using func_t = mlir::LogicalResult(*)(plier::PyCallOp, llvm::StringRef, llvm::ArrayRef<mlir::Value>, mlir::PatternRewriter&);
    using resolver_t = std::pair<check_t, func_t>;

    CallOpLowering(mlir::TypeConverter &/*typeConverter*/,
                   mlir::MLIRContext *context,
                   llvm::ArrayRef<resolver_t> resolvers):
        OpRewritePattern(context), resolvers(resolvers) {}

    mlir::LogicalResult matchAndRewrite(
        plier::PyCallOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto operands = op.getOperands();
        if (operands.empty())
        {
            return mlir::failure();
        }
        auto func_type = operands[0].getType();
        if (!func_type.isa<plier::PyType>())
        {
            return mlir::failure();
        }
        auto name = func_type.cast<plier::PyType>().getName();
        llvm::SmallVector<mlir::Type, 8> arg_types;
        llvm::SmallVector<mlir::Value, 8> args;
        if (name.consume_front("Function(") && name.consume_back(")"))
        {
            llvm::copy(llvm::drop_begin(op.getOperandTypes(), 1), std::back_inserter(arg_types));
            llvm::copy(llvm::drop_begin(op.getOperands(), 1), std::back_inserter(args));
            // TODO kwargs
        }
        else if (name.consume_front("BoundFunction(") && name.consume_back(")"))
        {
            auto getattr = mlir::dyn_cast<plier::GetattrOp>(operands[0].getDefiningOp());
            if (!getattr)
            {
                return mlir::failure();
            }
            arg_types.push_back(getattr.getOperand().getType());
            args.push_back(getattr.getOperand());
            llvm::copy(llvm::drop_begin(op.getOperandTypes(), 1), std::back_inserter(arg_types));
            llvm::copy(llvm::drop_begin(op.getOperands(), 1), std::back_inserter(args));
            name = extract_bound_func_name(name);
            // TODO kwargs
        }
        else
        {
            return mlir::failure();
        }
        for (auto& c : resolvers)
        {
            if (mlir::succeeded(c.first(name, arg_types)))
            {
                return c.second(op, name, args, rewriter);
            }
        }

        return mlir::failure();
    }

private:
    llvm::ArrayRef<resolver_t> resolvers;
};

struct PlierToLinalgPass :
    public mlir::PassWrapper<PlierToLinalgPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
        registry.insert<mlir::StandardOpsDialect>();
    }

    void runOnOperation() override;
};

void PlierToLinalgPass::runOnOperation()
{
    mlir::TypeConverter type_converter;
    // Convert unknown types to itself
    type_converter.addConversion([](mlir::Type type) { return type; });
    populate_std_type_converter(type_converter);
    type_converter.addConversion([&](plier::PyType type)->llvm::Optional<mlir::Type>
    {
        auto ret =  map_plier_type(type_converter, type);
        if (!ret)
        {
            return llvm::None;
        }
        return ret;
    });

    mlir::OwningRewritePatternList patterns;
    patterns.insert<
        FuncOpSignatureConversion
        >(type_converter, &getContext());

    patterns.insert<
        CallOpLowering
        >(type_converter, &getContext(), llvm::None);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
}

void populate_plier_to_linalg_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());
}
}

void register_plier_to_linalg_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_linalg_pipeline_name(), {plier_to_std_pipeline_name()}, {stage.end}, &populate_plier_to_linalg_pipeline);
    });
}

llvm::StringRef plier_to_linalg_pipeline_name()
{
    return "plier_to_linalg";
}
