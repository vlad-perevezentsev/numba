#include "pipelines/plier_to_linalg.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include "plier/dialect.hpp"

#include "pipelines/plier_to_std.hpp"
#include "rewrites/call_lowering.hpp"
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

mlir::LogicalResult numpy_rewrite(
    plier::PyCallOp op, llvm::StringRef name, llvm::ArrayRef<mlir::Value> args,
    mlir::PatternRewriter& rewriter)
{
    if (name == "<ufunc 'add'>" && check_numpy_args(args, 2))
    {
        mlir::Value inputs[] = { args[0], args[1] };
        auto elem_type = args[0].getType().cast<mlir::MemRefType>().getElementType();
        mlir::Type res_type = mlir::MemRefType::get({-1}, elem_type);
        auto loc = op.getLoc();
        mlir::Value size = rewriter.create<mlir::DimOp>(loc, args[0], 0);
        mlir::Value outputs[] = { rewriter.create<mlir::AllocOp>(loc, res_type, size) };
        mlir::AffineMap map[] = {
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
        };
        mlir::StringRef iterators[] = { "parallel" };

//        mlir::Value size = rewriter.create<mlir::DimOp>(loc, args[0], 0);
//        mlir::Value init = rewriter.create<mlir::DynamicTensorFromElementsOp>(
//            loc,
//            res_type,
//            mlir::ValueRange(size),
//            [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args)
//            {
//                assert(args.size() == 1);
//                auto val = builder.create<mlir::ConstantOp>(loc, mlir::IntegerAttr::get(elem_type, 0));
//                builder.create<mlir::YieldOp>(loc, val);
//            });

        auto body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args)
        {
            assert(args.size() == 3);
            mlir::Value res = builder.create<mlir::AddIOp>(loc, args[0], args[1]);
            builder.create<mlir::linalg::YieldOp>(loc, res);
        };
        rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::ValueRange(inputs),
            mlir::ValueRange(outputs),
            llvm::makeArrayRef(map),
            llvm::makeArrayRef(iterators),
            body);
        rewriter.replaceOp(op, outputs[0]);
        return mlir::success();
    }
    if (name == "array.sum" && check_numpy_args(args, 1))
    {
        mlir::Value inputs[] = { args[0] };
    //    auto elem_type = inputs[0].getType().cast<mlir::MemRefType>().getElementType();
        auto elem_type = mlir::IntegerType::get(64, op.getContext());
        auto res_type = mlir::MemRefType::get({}, elem_type);
        auto loc = op.getLoc();
        mlir::Value outputs[] = { rewriter.create<mlir::AllocaOp>(loc, res_type) };
        auto zero = rewriter.create<mlir::ConstantOp>(loc, get_zero(elem_type));
        rewriter.create<mlir::StoreOp>(loc, zero, outputs[0]);
        mlir::AffineMap map[] = {
            mlir::AffineMap::getMultiDimIdentityMap(1, op.getContext()),
            mlir::AffineMap::get(1, 0, op.getContext()),
        };
        mlir::StringRef iterators[] = { "reduction" };
        auto body = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args)
        {
            assert(args.size() == 2);
            auto val = builder.create<mlir::SignExtendIOp>(loc, args[0], elem_type);
            mlir::Value res = builder.create<mlir::AddIOp>(loc, val, args[1]);
            builder.create<mlir::linalg::YieldOp>(loc, res);
        };
        rewriter.create<mlir::linalg::GenericOp>(
            loc,
            mlir::ValueRange(inputs),
            mlir::ValueRange(outputs),
            llvm::makeArrayRef(map),
            llvm::makeArrayRef(iterators),
            body);
        mlir::Value res = rewriter.create<mlir::LoadOp>(loc, outputs[0]);
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
        if (!val.getType().template isa<mlir::MemRefType>())
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
        mlir::Value res = rewriter.create<mlir::LoadOp>(loc, val, index);
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
        >(type_converter, &getContext(), &numpy_rewrite);

    patterns.insert<
        GetitemOpLowering<plier::GetItemOp>,
        GetitemOpLowering<plier::StaticGetItemOp>
        >(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
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

    patterns.insert<mlir::linalg::LinalgLoweringPattern<mlir::linalg::GenericOp>>
        (&getContext(), mlir::linalg::LinalgLoweringType::ParallelLoops);


    (void)mlir::applyPatternsAndFoldGreedily(getOperation(), patterns);
}

void populate_plier_to_linalg_pipeline(mlir::OpPassManager& pm)
{
    pm.addPass(std::make_unique<PlierToLinalgPass>());
    pm.addPass(std::make_unique<LowerLinalgPass>());
    pm.addPass(mlir::createLowerToCFGPass());
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
