#include "pipelines/plier_to_linalg.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
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
    add_pipeline_jump_marker(get_module(op), marker);
}

mlir::LogicalResult numpy_rewrite(
    plier::PyCallOp op, llvm::StringRef name, llvm::ArrayRef<mlir::Value> args,
    mlir::PatternRewriter& rewriter)
{
    if (name == "<ufunc 'add'>" && check_numpy_args(args, 2))
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
            mlir::ValueRange(), // init
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
        auto elem_type = mlir::IntegerType::get(64, op.getContext());
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
            mlir::ValueRange(), // outputs
            mlir::ValueRange(init),
            llvm::makeArrayRef(map),
            llvm::makeArrayRef(iterators),
            body).getResult(0);
        mlir::Value index = rewriter.create<mlir::ConstantIndexOp>(loc, 0);
        mlir::Value res = rewriter.create<mlir::ExtractElementOp>(loc, val, index);
        rewriter.replaceOp(op, res);
        return mlir::success();
    }
    if (name == "<built-in function len>" && check_numpy_args(args, 1))
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
            res = rewriter.create<mlir::ExtractElementOp>(loc, val, index);
        }
        else
        {
            llvm_unreachable("Invalid getitem");
        }
        rewriter.replaceOp(op, res);
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
        GetitemOpLowering<plier::StaticGetItemOp>
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

void populate_plier_to_linalg_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());

    pm.addPass(mlir::createLinalgFusionOfTensorOpsPass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createStdBufferizePass());
    pm.addPass(mlir::createFuncBufferizePass());

    pm.addNestedPass<mlir::FuncOp>(mlir::createPromoteBuffersToStackPass(1024));
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferLoopHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCopyRemovalPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());

    pm.addPass(std::make_unique<LowerLinalgPass>());
}
}

void register_plier_to_linalg_pipeline(PipelineRegistry& registry)
{
    registry.register_pipeline([](auto sink)
    {
        auto stage = get_high_lowering_stage();
        sink(plier_to_linalg_pipeline_name(), {plier_to_std_pipeline_name()}, {stage.end}, {plier_to_std_pipeline_name()}, &populate_plier_to_linalg_pipeline);
    });
}

llvm::StringRef plier_to_linalg_pipeline_name()
{
    return "plier_to_linalg";
}
