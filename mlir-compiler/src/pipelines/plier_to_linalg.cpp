#include "pipelines/plier_to_linalg.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "plier/dialect.hpp"

#include "pipelines/plier_to_std.hpp"
#include "transforms/pipeline_utils.hpp"
#include "rewrites/call_lowering.hpp"
#include "rewrites/cast_lowering.hpp"
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
//            return mlir::MemRefType::get(shape, type);
            return mlir::RankedTensorType::get(shape, type);
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

bool check_numpy_args(llvm::ArrayRef<mlir::Value> args, unsigned expected_count)
{
    if (args.size() != expected_count)
    {
        return false;
    }
    for (auto arg : args)
    {
        auto type = arg.getType();
        if (!type.isa<mlir::MemRefType>() && !type.isa<mlir::TensorType>())
        {
            return false;
        }
    }
    return true;
}

mlir::Attribute get_zero(mlir::Type type)
{
    assert(type);
    if (auto int_type = type.dyn_cast<mlir::IntegerType>())
    {
        return mlir::IntegerAttr::get(type, 0);
    }
    if (auto float_type = type.dyn_cast<mlir::FloatType>())
    {
        return mlir::FloatAttr::get(type, 0.0);
    }
    llvm_unreachable("get_zero: usupported type");
}

mlir::Type get_elem_type(mlir::Type type)
{
    if (auto memref = type.dyn_cast<mlir::MemRefType>())
    {
        return memref.getElementType();
    }
    if (auto tensor = type.dyn_cast<mlir::TensorType>())
    {
        return tensor.getElementType();
    }
    llvm_unreachable("get_elem_type: unknown type");
}

void rerun_std_pipeline(mlir::Operation* op)
{
    assert(nullptr != op);
    auto marker = mlir::StringAttr::get(plier_to_std_pipeline_name(), op->getContext());
    add_pipeline_jump_marker(op->getParentOfType<mlir::ModuleOp>(), marker);
}

mlir::LogicalResult numpy_rewrite(
    plier::PyCallOp op, llvm::StringRef name, llvm::ArrayRef<mlir::Value> args,
    mlir::PatternRewriter& rewriter)
{
    if (name == "numpy.add" && check_numpy_args(args, 2))
    {
        auto loc = op.getLoc();
        mlir::Value inputs[] = { args[0], args[1] };
        auto elem_type = get_elem_type(args[0].getType());
        mlir::Type res_type = mlir::RankedTensorType::get(-1, elem_type);
        mlir::AffineMap map[] = {
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
        };
        mlir::StringRef iterators[] = { "parallel" };

        auto body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args)
        {
            assert(args.size() == 2);
            mlir::Value res = builder.create<mlir::AddIOp>(loc, args[0], args[1]);
            builder.create<mlir::linalg::YieldOp>(loc, res);
        };
        auto res = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::TypeRange(res_type),
            mlir::ValueRange(inputs),
            mlir::ValueRange(), // outputs
            llvm::makeArrayRef(map),
            llvm::makeArrayRef(iterators),
            body).getResult(0);
        rewriter.replaceOp(op, res);
        return mlir::success();
    }
    if (name == "array.sum" && check_numpy_args(args, 1))
    {
        auto loc = op.getLoc();
        mlir::Value inputs[] = { args[0] };
        auto elem_type = mlir::IntegerType::get(op.getContext(), 64);
        auto res_type = mlir::RankedTensorType::get(1, elem_type);
        mlir::Value zero = rewriter.create<mlir::ConstantOp>(loc, get_zero(elem_type));
        mlir::Value init = rewriter.create<mlir::TensorFromElementsOp>(loc, zero);
        mlir::AffineMap map[] = {
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
            mlir::AffineMap::get(1, 0, mlir::getAffineConstantExpr(0, op.getContext())),
        };
        mlir::StringRef iterators[] = { "reduction" };
        auto body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args)
        {
            assert(args.size() == 2);
            auto val = builder.create<mlir::SignExtendIOp>(loc, args[0], elem_type);
            mlir::Value res = builder.create<mlir::AddIOp>(loc, val, args[1]);
            builder.create<mlir::linalg::YieldOp>(loc, res);
        };
        auto val = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::TypeRange(res_type),
            mlir::ValueRange(inputs),
            mlir::ValueRange(init), // outputs
            llvm::makeArrayRef(map),
            llvm::makeArrayRef(iterators),
            body).getResult(0);
        mlir::Value index = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
        mlir::Value res = rewriter.create<mlir::tensor::ExtractOp>(loc, val, index);
        rewriter.replaceOp(op, res);
        return mlir::success();
    }
    if (name == "len" && check_numpy_args(args, 1))
    {
        auto loc = op.getLoc();
        mlir::Value dim = rewriter.create<mlir::DimOp>(loc, args[0], 0);
        mlir::Value res = rewriter.create<plier::CastOp>(loc, op.getType(), dim);
        rerun_std_pipeline(op);
        rewriter.replaceOp(op, res);
        return mlir::success();
    }
    return mlir::failure();
}

template<typename T>
struct GetitemOpLowering : public mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        T op, mlir::PatternRewriter &rewriter) const override
    {
        assert(op.getNumOperands() == 2);
        auto val = op.getOperand(0);
        auto index = op.getOperand(1);
        auto type = val.getType();
        bool is_memref = type.template isa<mlir::MemRefType>();
        bool is_tensor = type.template isa<mlir::TensorType>();
        if (!is_memref && !is_tensor)
        {
            return mlir::failure();
        }
        if (!index.getType().template isa<mlir::IndexType>() &&
            !index.getType().template isa<mlir::IntegerType>())
        {
            return mlir::failure();
        }
        auto loc = op.getLoc();
        if (index.getType().template isa<mlir::IntegerType>())
        {
            index = rewriter.create<mlir::IndexCastOp>(loc, index, mlir::IndexType::get(op.getContext()));
        }
        mlir::Value res;
        if (is_memref)
        {
            res = rewriter.create<mlir::LoadOp>(loc, val, index);
        }
        else if (is_tensor)
        {
            res = rewriter.create<mlir::tensor::ExtractOp>(loc, val, index);
        }
        else
        {
            llvm_unreachable("Invalid getitem");
        }
        rerun_std_pipeline(op);
        rewriter.replaceOp(op, res);
        return mlir::success();
    }
};

bool can_replace_ssa(mlir::Operation* op)
{
    assert(nullptr != op);
    if (op->getParentRegion()->getBlocks().size() != 1)
    {
        return false;
    }
    auto parent = op->getParentOp();
    if (mlir::isa<mlir::FuncOp>(parent))
    {
        return true;
    }
    return false;
//    return can_replace_ssa(parent);
}

bool replace_ssa_in_block(mlir::Value value, mlir::Value new_value, mlir::PatternRewriter &rewriter)
{
    auto new_op = new_value.getDefiningOp();
    assert(nullptr != new_op);
    auto block = new_op->getBlock();
    bool changed = false;
    for (auto user : llvm::make_early_inc_range(value.getUsers()))
    {
        if (auto op = block->findAncestorOpInBlock(*user))
        {
            if (op != new_op && new_op->isBeforeInBlock(op))
            {
                rewriter.updateRootInPlace(user, [&]()
                {
                    for (auto it2 : llvm::enumerate(user->getOperands()))
                    {
                        if (it2.value() == value)
                        {
                            user->setOperand(static_cast<unsigned>(it2.index()), new_value);
                            break;
                        }
                    }
                });
                changed = true;
            }
        }
    }
    return changed;
}

bool replace_ssa_value(mlir::Value value, mlir::Value new_value, mlir::PatternRewriter &rewriter)
{
    bool changed = replace_ssa_in_block(value, new_value, rewriter);
    auto parent = new_value.getDefiningOp()->getParentOp();
    if (auto func = mlir::dyn_cast<mlir::FuncOp>(parent))
    {
        // TODO update return
        return changed;
    }
    llvm_unreachable("Unhandled parent op");
}

mlir::Value index_cast(mlir::Value value, mlir::Location loc, mlir::OpBuilder& builder)
{
    if (!value.getType().isa<mlir::IndexType>())
    {
        auto index_type = mlir::IndexType::get(value.getContext());
        auto res = builder.create<plier::CastOp>(loc, index_type, value);
        rerun_std_pipeline(res);
        return res;
    }
    return value;
}

template<typename T>
struct SetitemOpLoweringSSA : public mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        T op, mlir::PatternRewriter &rewriter) const override
    {
        if (!can_replace_ssa(op))
        {
            return mlir::failure();
        }
        auto target = op.getOperand(0);
        auto index = op.getOperand(1);
        auto value = op.getOperand(2);
        auto target_type = target.getType().template dyn_cast<mlir::RankedTensorType>();
        if (!target_type)
        {
            return mlir::failure();
        }
        auto elem_type = target_type.getElementType();
        auto loc = op.getLoc();
        if (value.getType() != elem_type)
        {
            // TODO
            value = rewriter.create<plier::CastOp>(loc, elem_type, value);
            rerun_std_pipeline(op);
//            return mlir::failure();
        }

        auto new_tensor = rewriter.create<mlir::TensorFromElementsOp>(loc, value);
        auto new_index = index_cast(index, loc, rewriter);
        mlir::Value one = rewriter.create<mlir::ConstantIndexOp>(loc, 1);
        auto new_value = rewriter.create<mlir::SubTensorInsertOp>(loc, new_tensor, target, new_index, one, one);
        replace_ssa_value(target, new_value, rewriter);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct PlierToLinalgPass :
    public mlir::PassWrapper<PlierToLinalgPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<plier::PlierDialect>();
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
    }

    void runOnOperation() override;
};

template<typename T>
struct SetitemOpLowering : public mlir::OpRewritePattern<T>
{
    using mlir::OpRewritePattern<T>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        T op, mlir::PatternRewriter &rewriter) const override
    {
        auto get_target_type = [&]()
        {
            return op.getOperand(0).getType();
        };

        if (auto target_type = get_target_type().template dyn_cast<mlir::RankedTensorType>())
        {
            auto target = op.getOperand(0);
            mlir::OpBuilder::InsertionGuard g(rewriter);
            if (auto parent_op = target.getDefiningOp())
            {
                rewriter.setInsertionPoint(parent_op);
            }
            else
            {
                rewriter.setInsertionPointToStart(target.getParentBlock());
            }
            auto memref_type = mlir::MemRefType::get(target_type.getShape(), target_type.getElementType());
            auto memref = rewriter.create<mlir::TensorToMemrefOp>(target.getLoc(), memref_type, target);
            for (auto& use : llvm::make_early_inc_range(target.getUses()))
            {
                auto use_op = use.getOwner();
                assert(nullptr != use_op);
                if (use_op != memref)
                {
                    if (mlir::isa<plier::SetItemOp>(use_op))
                    {
                        use_op->setOperand(use.getOperandNumber(), memref);
                    }
                    else
                    {
                        rewriter.setInsertionPoint(use_op);
                        auto new_val = rewriter.create<mlir::TensorLoadOp>(use_op->getLoc(), memref);
                        rewriter.updateRootInPlace(use_op, [&]()
                        {
                            use_op->setOperand(use.getOperandNumber(), new_val);
                        });
                    }
                }
            }
        }
        else if (get_target_type().template isa<mlir::MemRefType>())
        {
            // nothing
        }
        else
        {
            return mlir::failure();
        }
        auto target = op.getOperand(0);
        auto index = op.getOperand(1);
        auto value = op.getOperand(2);
        auto loc = op.getLoc();
        auto ind = index_cast(index, loc, rewriter);
        auto elem_type = target.getType().template cast<mlir::MemRefType>().getElementType();
        if (value.getType() != elem_type)
        {
            // TODO
            value = rewriter.create<plier::CastOp>(loc, elem_type, value);
            rerun_std_pipeline(op);
        }
        auto store = rewriter.create<mlir::StoreOp>(loc, value, target, ind);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

void PlierToLinalgPass::runOnOperation()
{
    mlir::TypeConverter type_converter;
    // Convert unknown types to itself
    type_converter.addConversion([](mlir::Type type) { return type; });
    populate_std_type_converter(getContext(), type_converter);
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
        FuncOpSignatureConversion,
        CastOpLowering
        >(type_converter, &getContext());

    patterns.insert<
        CallOpLowering
        >(type_converter, &getContext(), &numpy_rewrite);

    patterns.insert<
        GetitemOpLowering<plier::GetItemOp>,
        GetitemOpLowering<plier::StaticGetItemOp>,
        SetitemOpLowering<plier::SetItemOp>
        >(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

struct LowerLinalgPass :
    public mlir::PassWrapper<LowerLinalgPass, mlir::OperationPass<mlir::ModuleOp>>
{
    virtual void getDependentDialects(
        mlir::DialectRegistry &registry) const override
    {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::linalg::LinalgDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::AffineDialect>();
    }

    void runOnOperation() override;
};

void LowerLinalgPass::runOnOperation()
{
    mlir::OwningRewritePatternList patterns;

    patterns.insert<
        mlir::linalg::LinalgLoweringPattern<mlir::linalg::GenericOp>,
        mlir::linalg::LinalgLoweringPattern<mlir::linalg::CopyOp>
        >(&getContext(), mlir::linalg::LinalgLoweringType::ParallelLoops);


    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void populate_plier_to_linalg_gen_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());
}

void populate_plier_to_linalg_opt_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(mlir::createLinalgFusionOfTensorOpsPass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createTensorBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createStdBufferizePass());
    pm.addPass(mlir::createFuncBufferizePass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createPromoteBuffersToStackPass(1024));
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferLoopHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCopyRemovalPass());

    pm.addPass(std::make_unique<LowerLinalgPass>());
}
}

void register_plier_to_linalg_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_linalg_gen_pipeline_name(), {plier_to_std_pipeline_name()}, {plier_to_linalg_opt_pipeline_name()}, {plier_to_std_pipeline_name()}, &populate_plier_to_linalg_gen_pipeline);
        sink(plier_to_linalg_opt_pipeline_name(), {plier_to_linalg_gen_pipeline_name()}, {stage.end}, {}, &populate_plier_to_linalg_opt_pipeline);
    });
}

llvm::StringRef plier_to_linalg_gen_pipeline_name()
{
    return "plier_to_linalg_gen";
}

llvm::StringRef plier_to_linalg_opt_pipeline_name()
{
    return "plier_to_linalg_opt";
}
