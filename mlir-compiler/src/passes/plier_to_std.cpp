#include "plier_to_std.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "plier/dialect.hpp"

namespace
{
mlir::Type map_int_type(plier::PyType type)
{
    auto name = type.getName();
    unsigned num_bits = 0;
    if (name.consume_front("int") &&
        !name.consumeInteger<unsigned>(10, num_bits) && name.empty())
    {
        return mlir::IntegerType::get(num_bits, type.getContext());
    }
    return {};
}

mlir::Type map_plier_type(mlir::Type type)
{
    if (!type.isa<plier::PyType>())
    {
        return {};
    }
    auto ptype = type.cast<plier::PyType>();
    using func_t = mlir::Type(*)(plier::PyType);
    const func_t handlers[] = {
        &map_int_type
    };
    for (auto h : handlers)
    {
        auto t = h(ptype);
        if (t != mlir::Type())
        {
            return t;
        }
    }
    return {};
}

bool is_supported_type(mlir::Type type)
{
    return type.isIntOrFloat();
}

mlir::Type map_type(mlir::Type type)
{
    auto new_type = is_supported_type(type) ? type : map_plier_type(type);
    return mlir::Type() == new_type ? type : new_type;
};

void convertFuncArgs(mlir::FuncOp func)
{
    llvm::SmallVector<mlir::Type, 8> new_arg_types;
    new_arg_types.reserve(func.getNumArguments());
    for (auto arg_type : func.getArgumentTypes())
    {
        new_arg_types.push_back(map_type(arg_type));
    }
    auto res_type = map_type(func.getType().getResult(0));
    auto func_type = mlir::FunctionType::get(new_arg_types, res_type, func.getContext());
    func.setType(func_type);
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
    {
        func.front().getArgument(i).setType(new_arg_types[i]);
    }
}

template<typename T>
void replace_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type new_type)
{
    assert(nullptr != op);
    rewriter.replaceOpWithNewOp<T>(op, new_type, op->getOperands());
}

bool is_int(mlir::Type type)
{
    return type.isa<mlir::IntegerType>();
}

bool is_float(mlir::Type type)
{
    return type.isa<mlir::FloatType>();
}

struct ConstOpLowering : public mlir::OpRewritePattern<plier::ConstOp>
{
    using mlir::OpRewritePattern<plier::ConstOp>::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(plier::ConstOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        auto value = op.val();
        if (!is_supported_type(value.getType()))
        {
            return mlir::failure();
        }
        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, value.getType(), value);
        return mlir::success();
    }
};

struct BinOpLowering : public mlir::OpRewritePattern<plier::BinOp>
{
    using mlir::OpRewritePattern<plier::BinOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(plier::BinOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        assert(op.getNumOperands() == 2);
        auto type0 = op.getOperand(0).getType();
        auto type1 = op.getOperand(1).getType();
        if (type0 != type1 || !is_supported_type(type0) || !is_supported_type(type1))
        {
            return mlir::failure();
        }

        using func_t = void(*)(mlir::Operation*, mlir::PatternRewriter&, mlir::Type);
        struct OpDesc
        {
            llvm::StringRef type;
            func_t iop;
            func_t fop;
        };

        const OpDesc handlers[] = {
            {"+", &replace_op<mlir::AddIOp>, &replace_op<mlir::AddFOp>}
        };

        auto find_handler = [&]()->const OpDesc&
        {
            for (auto& h : handlers)
            {
                if (h.type == op.op())
                {
                    return h;
                }
            }
            llvm_unreachable("Unhandled op type");
        };

        if (is_int(type0))
        {
            find_handler().iop(op, rewriter, type0);
        }
        else if (is_float(type0))
        {
            find_handler().fop(op, rewriter, type0);
        }
        else
        {
            llvm_unreachable("Unhandled arg type");
        }

        return mlir::success();
    }
};

using namespace mlir;
struct FuncOpSignatureConversion : public OpConversionPattern<FuncOp> {
  FuncOpSignatureConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(converter, ctx) {}

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    convertFuncArgs(funcOp);
    rewriter.updateRootInPlace(funcOp, [&] {}); // HACK
    return success();
  }
};

struct PlierToStdPass :
    public mlir::PassWrapper<PlierToStdPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
    }

    void runOnOperation() override;
};

void PlierToStdPass::runOnOperation()
{
    mlir::TypeConverter type_converter;
    type_converter.addConversion([](plier::Type type)->llvm::Optional<mlir::Type>
    {
        return map_plier_type(type);
    });

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::StandardOpsDialect>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<ConstOpLowering, BinOpLowering>(&getContext());
    patterns.insert<FuncOpSignatureConversion>(&getContext(), type_converter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
    {
        signalPassFailure();
    }
}

}

std::unique_ptr<mlir::Pass> createPlierToStdPass()
{
    return std::make_unique<PlierToStdPass>();
}
