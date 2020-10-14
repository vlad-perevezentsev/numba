#include "passes/plier_to_std.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/TypeSwitch.h>

#include "plier/dialect.hpp"

#include "base_pipeline.hpp"
#include "pipeline_registry.hpp"

namespace
{
mlir::Type map_int_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    unsigned num_bits = 0;
    if (name.consume_front("int") &&
        !name.consumeInteger<unsigned>(10, num_bits))
    {
        return mlir::IntegerType::get(num_bits, &ctx);
    }
    return nullptr;
}

mlir::Type map_int_literal_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    unsigned dummy = 0;
    if (name.consume_front("Literal[int](") &&
        !name.consumeInteger<unsigned>(10, dummy) && name.consume_front(")"))
    {
        return mlir::IntegerType::get(64, &ctx); // TODO
    }
    return nullptr;
}

mlir::Type map_bool_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    if (name.consume_front("bool"))
    {
        return mlir::IntegerType::get(1, &ctx);
    }
    return nullptr;
}

mlir::Type map_float_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    unsigned num_bits = 0;
    if (name.consume_front("float") &&
        !name.consumeInteger<unsigned>(10, num_bits))
    {
        switch(num_bits)
        {
        case 64: return mlir::Float64Type::get(&ctx);
        case 32: return mlir::Float32Type::get(&ctx);
        case 16: return mlir::Float16Type::get(&ctx);
        }
    }
    return nullptr;
}

mlir::Type map_plier_type_name(mlir::MLIRContext& ctx, llvm::StringRef& name);

mlir::Type map_pair_type(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    if (!name.consume_front("pair<"))
    {
        return nullptr;
    }
    auto first = map_plier_type_name(ctx, name);
    if (!static_cast<bool>(first))
    {
        return nullptr;
    }
    if (!name.consume_front(", "))
    {
        return nullptr;
    }
    auto second = map_plier_type_name(ctx, name);
    if (!static_cast<bool>(second))
    {
        return nullptr;
    }
    if (!name.consume_front(">"))
    {
        return nullptr;
    }
    return mlir::TupleType::get({first, second}, &ctx);
}

mlir::Type map_plier_type_name(mlir::MLIRContext& ctx, llvm::StringRef& name)
{
    using func_t = mlir::Type(*)(mlir::MLIRContext& ctx, llvm::StringRef& name);
    const func_t handlers[] = {
        &map_int_type,
        &map_int_literal_type,
        &map_bool_type,
        &map_float_type,
        &map_pair_type,
    };
    for (auto h : handlers)
    {
        auto temp_name = name;
        auto t = h(ctx, temp_name);
        if (static_cast<bool>(t))
        {
            name = temp_name;
            return t;
        }
    }
    return nullptr;
}

mlir::Type map_plier_type(mlir::Type type)
{
    if (!type.isa<plier::PyType>())
    {
        return type;
    }
    auto name = type.cast<plier::PyType>().getName();
    return map_plier_type_name(*type.getContext(), name);
}

bool is_supported_type(mlir::Type type)
{
    return type.isIntOrFloat();
}

mlir::Type map_type(mlir::Type type)
{
    auto new_type = is_supported_type(type) ? type : map_plier_type(type);
    return static_cast<bool>(new_type) ? new_type : type;
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
void replace_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type new_type, mlir::ValueRange operands)
{
    assert(nullptr != op);
    rewriter.replaceOpWithNewOp<T>(op, new_type, operands);
}

template<typename T, uint64_t Pred>
void replace_cmp_op(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::Type /*new_type*/, mlir::ValueRange operands)
{
    assert(nullptr != op);
    auto pred_attr = mlir::IntegerAttr::get(mlir::IntegerType::get(64, op->getContext()), Pred);
    mlir::Type new_type = mlir::IntegerType::get(1, op->getContext());
    rewriter.replaceOpWithNewOp<T>(op, new_type, pred_attr, operands[0], operands[1]);
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

mlir::Type coerce(mlir::Type type0, mlir::Type type1)
{
    // TODO: proper rules
    assert(type0 != type1);
    auto get_bits_count = [](mlir::Type type)->unsigned
    {
        if (type.isa<mlir::IntegerType>())
        {
            return type.cast<mlir::IntegerType>().getWidth();
        }
        if (type.isa<mlir::Float16Type>())
        {
            return 11;
        }
        if (type.isa<mlir::Float32Type>())
        {
            return 24;
        }
        if (type.isa<mlir::Float64Type>())
        {
            return 53;
        }
        llvm_unreachable("Unhandled type");
    };
    auto f0 = is_float(type0);
    auto f1 = is_float(type1);
    if (f0 && !f1)
    {
        return type0;
    }
    if (!f0 && f1)
    {
        return type1;
    }
    return get_bits_count(type0) < get_bits_count(type1) ? type1 : type0;
}

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

template <bool Signed>
mlir::Value int_float_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    using T = std::conditional_t<Signed, mlir::SIToFPOp, mlir::UIToFPOp>;
    return rewriter.create<T>(val.getLoc(), val, dst_type);
}

template <bool Signed>
mlir::Value float_int_cast(mlir::Type dst_type, mlir::Value val, mlir::PatternRewriter& rewriter)
{
    using T = std::conditional_t<Signed, mlir::FPToSIOp, mlir::FPToUIOp>;
    return rewriter.create<T>(val.getLoc(), val, dst_type);
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
        {&is_int, &is_float, &int_float_cast<true>},
        {&is_float, &is_int, &float_int_cast<true>},
    };

    for (auto& h : handlers)
    {
        if (h.src(src_type) && h.dst(dst_type))
        {
            return h.cast_op(dst_type, val, rewriter);
        }
    }

    llvm_unreachable("Unhandled cast");
}

struct BinOpLowering : public mlir::OpRewritePattern<plier::BinOp>
{
    using mlir::OpRewritePattern<plier::BinOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(plier::BinOp op,
                                        mlir::PatternRewriter& rewriter) const
    {
        assert(op.getNumOperands() == 2);
        auto type0 = op.getOperand(0).getType();
        auto type1 = op.getOperand(1).getType();
        if (!is_supported_type(type0) || !is_supported_type(type1))
        {
            return mlir::failure();
        }
        mlir::Type final_type;
        std::array<mlir::Value, 2> operands;
        if (type0 != type1)
        {
            final_type = coerce(type0, type1);
            operands = {do_cast(final_type, op.getOperand(0), rewriter),
                        do_cast(final_type, op.getOperand(1), rewriter)};
        }
        else
        {
            final_type = type0;
            operands = {op.getOperand(0), op.getOperand(1)};
        }
        assert(static_cast<bool>(final_type));

        using func_t = void(*)(mlir::Operation*, mlir::PatternRewriter&, mlir::Type, mlir::ValueRange);
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
            {">=", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::sge)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OGE)>},
            {"<", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::slt)>,
                  &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OLT)>},
            {"<=", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::sle)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OLE)>},
            {"!=", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::ne)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::ONE)>},
            {"==", &replace_cmp_op<mlir::CmpIOp, static_cast<uint64_t>(mlir::CmpIPredicate::eq)>,
                   &replace_cmp_op<mlir::CmpFOp, static_cast<uint64_t>(mlir::CmpFPredicate::OEQ)>},
        };

        using membptr_t = func_t OpDesc::*;
        auto call_handler = [&](membptr_t mem)
        {
            for (auto& h : handlers)
            {
                if (h.type == op.op())
                {
                    (h.*mem)(op, rewriter, final_type, operands);
                    return mlir::success();
                }
            }
            return mlir::failure();
        };


        if (is_int(final_type))
        {
            return call_handler(&OpDesc::iop);
        }
        else if (is_float(final_type))
        {
            return call_handler(&OpDesc::fop);
        }
        return mlir::failure();
    }
};

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

struct CastOpLowering : public mlir::OpRewritePattern<plier::CastOp>
{
    using mlir::OpRewritePattern<plier::CastOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(plier::CastOp op, mlir::PatternRewriter& rewriter) const override
    {
        auto src_type = op.getOperand().getType();
        auto dst_type = op.getType();
        if (is_supported_type(src_type) && is_supported_type(dst_type))
        {
            auto new_op = do_cast(dst_type, op.getOperand(), rewriter);
            rewriter.replaceOp(op, new_op);
            return mlir::success();
        }
        return mlir::failure();
    }
};

mlir::Operation* change_op_ret_type(mlir::Operation* op,
                                    mlir::PatternRewriter& rewriter,
                                    llvm::ArrayRef<mlir::Type> types)
{
    assert(nullptr != op);
    mlir::OperationState state(op->getLoc(), op->getName().getStringRef(),
                                   op->getOperands(), types, op->getAttrs());
    return rewriter.createOperation(state);
}

struct ExpandTuples : public mlir::RewritePattern
{
    ExpandTuples(mlir::MLIRContext* ctx):
        RewritePattern(0, mlir::Pattern::MatchAnyOpTypeTag()),
        dialect(ctx->getLoadedDialect<plier::PlierDialect>())
    {
        assert(nullptr != dialect);
    }

    mlir::LogicalResult
    matchAndRewrite(plier::Operation* op, mlir::PatternRewriter& rewriter) const override
    {
        if (op->getResultTypes().size() != 1 ||
            !op->getResultTypes()[0].isa<mlir::TupleType>() ||
            (op->getDialect() != dialect))
        {
            return mlir::failure();
        }
        auto types = op->getResultTypes()[0].cast<mlir::TupleType>().getTypes();

        auto new_op = change_op_ret_type(op, rewriter, types);
        auto new_op_results = new_op->getResults();

        llvm::SmallVector<mlir::Operation*, 8> users(op->getUsers());
        llvm::SmallVector<mlir::Value, 8> new_operands;
        op->dump();
        for (auto user_op : users)
        {
            new_operands.clear();
            for (auto arg : user_op->getOperands())
            {
                if (arg.getDefiningOp() == op)
                {
                    std::copy(new_op_results.begin(), new_op_results.end(), std::back_inserter(new_operands));
                }
                else
                {
                    new_operands.push_back(arg);
                }
            }
            rewriter.updateRootInPlace(user_op, [&]()
            {
                user_op->setOperands(new_operands);
            });
        }
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    mlir::Dialect* dialect = nullptr;
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
                changed = changed || (new_type != type);
                new_types.push_back(new_type);
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
        registry.insert<plier::PlierDialect>();
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
                    CallOpLowering, CastOpLowering,
                    ExpandTuples>(&getContext());

    auto apply_conv = [&]()
    {
        (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
    };

    apply_conv();
}

void populate_plier_to_std_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(std::make_unique<PlierToStdPass>());
}
}


void register_plier_to_std_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink("plier_to_std", {stage.begin}, {stage.end}, &populate_plier_to_std_pipeline);
    });
}
