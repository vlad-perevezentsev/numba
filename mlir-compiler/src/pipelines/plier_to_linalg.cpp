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
