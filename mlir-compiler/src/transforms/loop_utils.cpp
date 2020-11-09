#include "transforms/loop_utils.hpp"

#include <llvm/ADT/SmallVector.h>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "plier/dialect.hpp"

namespace
{
template<typename Op>
Op get_next_op(llvm::iterator_range<mlir::Block::iterator>& iters)
{
    if (iters.empty())
    {
        return nullptr;
    }
    auto res = mlir::dyn_cast<Op>(iters.begin());
    if (res)
    {
        auto next = std::next(iters.begin());
        iters = {next, iters.end()};
    }
    return res;
}
}

mlir::LogicalResult lower_while_to_for(
    plier::GetiterOp getiter, mlir::PatternRewriter& builder,
    llvm::function_ref<std::tuple<mlir::Value,mlir::Value,mlir::Value>(mlir::OpBuilder&, mlir::Location)> get_bounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::Type, mlir::Value)> get_iter_val)
{
    llvm::SmallVector<mlir::scf::WhileOp, 4> to_process;
    for (auto user : getiter.getOperation()->getUsers())
    {
        if( auto while_op = mlir::dyn_cast<mlir::scf::WhileOp>(user->getParentOp()))
        {
            to_process.emplace_back(while_op);
        }
    }

    bool changed = false;
    for (auto while_op : to_process)
    {
        auto& before_block = while_op.before().front();
        auto iters = llvm::iterator_range<mlir::Block::iterator>(before_block);
        auto iternext = get_next_op<plier::IternextOp>(iters);
        auto pairfirst = get_next_op<plier::PairfirstOp>(iters);
        auto pairsecond = get_next_op<plier::PairsecondOp>(iters);
        while (get_next_op<plier::CastOp>(iters)) {} // skip casts
        auto before_term = get_next_op<mlir::scf::ConditionOp>(iters);

        auto skip_casts = [](mlir::Value op)
        {
            while (auto cast = mlir::dyn_cast_or_null<plier::CastOp>(op.getDefiningOp()))
            {
                op = cast.getOperand();
            }
            return op;
        };
        if (!iternext || !pairfirst || !pairsecond || !before_term ||
            skip_casts(before_term.condition()) != pairsecond)
        {
            continue;
        }

        auto& after_block = while_op.after().front();

        auto body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value iv, mlir::ValueRange iterargs)
        {
            mlir::BlockAndValueMapping mapper;
            assert(before_block.getNumArguments() == iterargs.size());
            assert(after_block.getNumArguments() == before_term.args().size());
            mapper.map(before_block.getArguments(), iterargs);
            for (auto it : llvm::zip(after_block.getArguments(), before_term.args()))
            {
                auto block_arg = std::get<0>(it);
                auto term_arg = std::get<1>(it);
                if (term_arg == pairfirst) // iter arg
                {
                    auto iter_val = get_iter_val(builder, loc, pairfirst.getType(), iv);
                    mapper.map(block_arg, iter_val);
                }
                else
                {
                    mapper.map(block_arg, mapper.lookupOrDefault(term_arg));
                }
            }

            for (auto& op : after_block) // with terminator
            {
                builder.clone(op, mapper);
            }
        };

        auto loc = getiter.getLoc();

        auto index_cast = [&](mlir::Value val)->mlir::Value
        {
            if (!val.getType().isa<mlir::IndexType>())
            {
                return builder.create<mlir::IndexCastOp>(loc, val, mlir::IndexType::get(val.getContext()));
            }
            return val;
        };

        auto bounds = get_bounds(builder, loc);
        auto lower_bound = index_cast(std::get<0>(bounds));
        auto upper_bound = index_cast(std::get<1>(bounds));
        auto step = index_cast(std::get<2>(bounds));

        builder.setInsertionPoint(while_op);
        auto loop_op = builder.create<mlir::scf::ForOp>(
            loc,
            lower_bound,
            upper_bound,
            step,
            while_op.getOperands(), // iterArgs
            body
            );

        assert(while_op.getNumResults() >= loop_op.getNumResults());
        builder.updateRootInPlace(while_op, [&]()
        {
            assert(while_op.getNumResults() == before_term.args().size());
            for (auto it : llvm::zip(while_op.getResults(), before_term.args()))
            {
                auto old_res = std::get<0>(it);
                auto operand = std::get<1>(it);
                for (auto it2 : llvm::enumerate(before_block.getArguments()))
                {
                    auto arg = it2.value();
                    if (arg == operand)
                    {
                        assert(it2.index() < loop_op.getNumResults());
                        auto new_res = loop_op.getResult(static_cast<unsigned>(it2.index()));
                        old_res.replaceAllUsesWith(new_res);
                    }
                }
            }
        });

        assert(while_op.getOperation()->getUsers().empty());
        builder.eraseOp(while_op);
        changed = true;
    }

    if (getiter.getOperation()->getUsers().empty())
    {
        builder.eraseOp(getiter);
        changed = true;
    }
    return mlir::success(changed);
}

