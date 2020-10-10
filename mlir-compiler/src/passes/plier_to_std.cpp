#include "passes/plier_to_std.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

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

mlir::Type map_int_literal_type(plier::PyType type)
{
    auto name = type.getName();
    unsigned dummy = 0;
    if (name.consume_front("Literal[int](") &&
        !name.consumeInteger<unsigned>(10, dummy) && name.consume_front(")")
        && name.empty())
    {
        return mlir::IntegerType::get(64, type.getContext()); // TODO
    }
    return {};
}

mlir::Type map_bool_type(plier::PyType type)
{
    auto name = type.getName();
    if (name == "bool")
    {
        return mlir::IntegerType::get(1, type.getContext());
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
        &map_int_type,
        &map_int_literal_type,
        &map_bool_type,
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

bool convert_func_sig(mlir::FuncOp func)
{
    llvm::SmallVector<mlir::Type, 8> new_arg_types;
    new_arg_types.reserve(func.getNumArguments());
    bool changed = false;
    for (auto arg_type : func.getArgumentTypes())
    {
        auto new_type = map_type(arg_type);
        changed = changed || (new_type != arg_type);
        new_arg_types.push_back(new_type);
    }

    auto res_type = func.getType().getResult(0);
    auto new_res_type = map_type(res_type);
    changed = changed || (res_type != new_res_type);
    if (changed)
    {
        auto func_type = mlir::FunctionType::get(new_arg_types, new_res_type, func.getContext());
        func.setType(func_type);
        for (unsigned i = 0; i < func.getNumArguments(); ++i)
        {
            func.front().getArgument(i).setType(new_arg_types[i]);
        }
    }
    for (auto& bb : llvm::make_range(++func.getBody().begin(),
                                     func.getBody().end()))
    {
        for (auto arg : bb.getArguments())
        {
            auto arg_type = arg.getType();
            auto new_type = map_type(arg_type);
            if (new_type != arg_type)
            {
                arg.setType(new_type);
                changed = true;
            }
        }
    }
    return changed;
}

template<typename T>
void replace_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type new_type)
{
    assert(nullptr != op);
    rewriter.replaceOpWithNewOp<T>(op, new_type, op->getOperands());
}

template<typename T, uint64_t Pred>
void replace_cmp_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type /*new_type*/)
{
    assert(nullptr != op);
    auto pred_attr = mlir::IntegerAttr::get(mlir::IntegerType::get(64, op->getContext()), Pred);
    mlir::Type new_type = mlir::IntegerType::get(1, op->getContext());
    rewriter.replaceOpWithNewOp<T>(op, new_type, pred_attr, op->getOperand(0), op->getOperand(1));
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
            {"+", &replace_op<mlir::AddIOp>, &replace_op<mlir::AddFOp>},
            {"-", &replace_op<mlir::SubIOp>, &replace_op<mlir::SubFOp>},
            {"*", &replace_op<mlir::MulIOp>, &replace_op<mlir::MulFOp>},

            {">", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::sgt)>,
                  &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OGT)>},
        };

        using membptr_t = func_t OpDesc::*;
        auto call_handler = [&](membptr_t mem)
        {
            for (auto& h : handlers)
            {
                if (h.type == op.op())
                {
                    (h.*mem)(op, rewriter, type0);
                    return mlir::success();
                }
            }
            return mlir::failure();
        };


        if (is_int(type0))
        {
            return call_handler(&OpDesc::iop);
        }
        else if (is_float(type0))
        {
            return call_handler(&OpDesc::fop);
        }
        return mlir::failure();
    }
};

template <bool Signed>
mlir::Value int_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    auto src_bits = val.getType().cast<mlir::IntegerType>().getWidth();
    auto dst_bits = dst_type.cast<mlir::IntegerType>().getWidth();
    assert(src_bits != dst_bits);
    if (dst_bits > src_bits)
    {
        using T = std::conditional_t<Signed, mlir::SignExtendIOp, mlir::ZeroExtendIOp>;
        return rewriter.create<T>(val.getLoc(), val, dst_type);
    }
    else
    {
        return rewriter.create<mlir::TruncateIOp>(val.getLoc(), val, dst_type);
    }
}

mlir::Value do_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    auto src_type = val.getType();
    if (src_type == dst_type)
    {
        return val;
    }

    struct Handler
    {
        using selector_t = bool(*)(mlir::Type);
        using cast_op_t = mlir::Value(*)(mlir::Type, mlir::Value, mlir::PatternRewriter&);
        selector_t src;
        selector_t dst;
        cast_op_t cast_op;
    };

    const Handler handlers[] = {
        {&is_int, &is_int, &int_cast<true>},
    };

    for (auto& h : handlers)
    {
        if (h.src(src_type) && h.dst(dst_type))
        {
            return h.cast_op(dst_type, val, rewriter);
        }
    }

    return nullptr;
}

mlir::LogicalResult lower_bool_cast(plier::PyCallOp op, mlir::PatternRewriter& rewriter)
{
    if (op.getNumOperands() != 2)
    {
        return mlir::failure();
    }
    auto val = op.getOperand(1);
    bool success = false;
    auto replace_op = [&](mlir::Value val)
    {
        assert(!success);
        if (val)
        {
            rewriter.replaceOp(op, val);
            success = true;
        }
    };
    auto src_type = val.getType();
    auto dst_type = mlir::IntegerType::get(1, op.getContext());
    mlir::TypeSwitch<mlir::Type>(src_type)
        .Case<mlir::IntegerType>([&](auto) { replace_op(do_cast(dst_type, val, rewriter)); });
    return mlir::success(success);
}

using call_lowerer_func_t = mlir::LogicalResult(*)(plier::PyCallOp, mlir::PatternRewriter&);
const constexpr std::pair<llvm::StringRef, call_lowerer_func_t> builtin_calls[] = {
    {"<class 'bool'>", &lower_bool_cast},
};

struct CallOpLowering : public mlir::OpRewritePattern<plier::PyCallOp>
{
    using mlir::OpRewritePattern<plier::PyCallOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(plier::PyCallOp op, mlir::PatternRewriter& rewriter) const override
    {
        if (op.getNumOperands() == 0)
        {
            return mlir::failure();
        }
        auto func_type = op.getOperand(0).getType();
        if (!func_type.isa<plier::PyType>())
        {
            return mlir::failure();
        }
        auto name = func_type.cast<plier::PyType>().getName();
        if (!name.consume_front("Function(") || !name.consume_back(")"))
        {
            return mlir::failure();
        }
        for (auto& c : builtin_calls)
        {
            if (c.first == name)
            {
                return c.second(op, rewriter);
            }
        }
        return mlir::failure();
    }
};


struct FuncOpSignatureConversion : public mlir::OpRewritePattern<mlir::FuncOp>
{
    FuncOpSignatureConversion(mlir::MLIRContext* ctx,
                              mlir::TypeConverter& /*converter*/)
        : OpRewritePattern(ctx) {}

    /// Hook for derived classes to implement combined matching and rewriting.
    mlir::LogicalResult
    matchAndRewrite(mlir::FuncOp funcOp, mlir::PatternRewriter &rewriter) const override
    {
        bool changed = convert_func_sig(funcOp);
        if (changed)
        {
            rewriter.updateRootInPlace(funcOp, [&] {}); // HACK
        }
        return mlir::success(changed);
    }
};

struct OpTypeConversion : public mlir::RewritePattern
{
    OpTypeConversion(mlir::MLIRContext* /*ctx*/,
                     mlir::TypeConverter& /*converter*/)
        : RewritePattern(0, mlir::Pattern::MatchAnyOpTypeTag()) {}

    /// Hook for derived classes to implement combined matching and rewriting.
    mlir::LogicalResult
    matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter &rewriter) const override
    {
        bool changed = false;
        llvm::SmallVector<mlir::Type, 8> new_types;
        for (auto type : op->getResultTypes())
        {
            if (auto new_type = map_plier_type(type))
            {
                new_types.push_back(new_type);
                changed = true;
            }
            else
            {
                new_types.push_back(type);
            }
        }

        if (changed)
        {
            rewriter.updateRootInPlace(op, [&]
            {
                for (unsigned i = 0; i < static_cast<unsigned>(new_types.size()); ++i)
                {
                    op->getResult(i).setType(new_types[i]);
                }
            });
        }
        return mlir::success(changed);
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

    mlir::OwningRewritePatternList patterns;
    patterns.insert<FuncOpSignatureConversion,
                    OpTypeConversion>(&getContext(), type_converter);
    patterns.insert<ConstOpLowering, BinOpLowering,
                    CallOpLowering>(&getContext());

    auto apply_conv = [&]()
    {
        return mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
    };

    if (mlir::failed(apply_conv()))
    {
        signalPassFailure();
        return;
    }
}

}

std::unique_ptr<mlir::Pass> createPlierToStdPass()
{
    return std::make_unique<PlierToStdPass>();
}
