#include "lowering.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include <pybind11/pybind11.h>

#include <llvm/IR/Type.h>

#include <mlir/IR/Module.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>

#include <mlir/Target/LLVMIR.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "type_parser.hpp"

#include <llvm/Bitcode/BitcodeWriter.h>

#include <iostream>

namespace py = pybind11;
namespace mllvm = mlir::LLVM;
namespace
{
[[noreturn]] void report_error(const llvm::Twine& msg)
{
    auto str = msg.str();
    throw std::exception(str.c_str());
}

std::string serialize_mod(const llvm::Module& mod)
{
    std::string ret;
    llvm::raw_string_ostream stream(ret);
//    mod.print(stream, nullptr);
    llvm::WriteBitcodeToFile(mod, stream);
    stream.flush();
    return ret;
}

template<typename T>
std::string to_str(T& obj)
{
    std::string ret;
    llvm::raw_string_ostream stream(ret);
    obj.print(stream);
    stream.flush();
    return ret;
}

mllvm::LLVMDialect& get_dialect(mlir::MLIRContext& ctx)
{
    auto dialect = ctx.getRegisteredDialect<mllvm::LLVMDialect>();
    assert(nullptr != dialect);
    return *dialect;
}

std::vector<std::pair<int, py::handle>> get_blocks(const py::object& func)
{
    std::vector<std::pair<int, py::handle>> ret;
    auto blocks = func.attr("blocks").cast<py::dict>();
    ret.reserve(blocks.size());
    for (auto it : blocks)
    {
        ret.push_back({it.first.cast<int>(), it.second});
    }
    return ret;
}

py::list get_body(const py::handle& block)
{
    return block.attr("body").cast<py::list>();
}

struct scoped_goto_block
{
    scoped_goto_block(mlir::OpBuilder& b, mlir::Block* new_block):
        builder(b),
        old_block(b.getBlock())
    {
        builder.setInsertionPointToEnd(new_block);
    }

    ~scoped_goto_block()
    {
        builder.setInsertionPointToEnd(old_block);
    }

    mlir::OpBuilder& builder;
    mlir::Block* old_block = nullptr;
};

struct inst_handles
{
    inst_handles()
    {
        auto mod = py::module::import("numba.core.ir");
        Assign = mod.attr("Assign");
        Del = mod.attr("Del");
        Return = mod.attr("Return");

        Arg = mod.attr("Arg");
        Const = mod.attr("Const");
        Global = mod.attr("Global");
        Expr = mod.attr("Expr");

        auto ops = py::module::import("operator");

        add = ops.attr("add");

        eq = ops.attr("eq");
        gt = ops.attr("gt");
    }

    py::handle Assign;
    py::handle Del;
    py::handle Return;

    py::handle Arg;
    py::handle Const;
    py::handle Global;
    py::handle Expr;

    py::handle add;

    py::handle eq;
    py::handle gt;
};

struct type_cache
{
    using Type = mllvm::LLVMType;

    Type get_type(mllvm::LLVMDialect& dialect, llvm::StringRef str)
    {
        assert(!str.empty());
        auto s = str.str();
        auto it = typemap.find(s);
        if (typemap.end() != it)
        {
            return it->second;
        }
        auto type = parse_type(dialect, str);
        typemap[s] = type;
        return type;
    }

private:
    std::unordered_map<std::string, Type> typemap;
};

struct lowerer
{
    lowerer():
        dialect(get_dialect(ctx)),
        builder(&ctx)
    {

    }

    py::bytes lower(const py::object& compilation_context, const py::object& func_ir)
    {
        auto mod = mlir::ModuleOp::create(builder.getUnknownLoc());
        auto typ = get_func_type(compilation_context["fntype"]);
        auto name = compilation_context["fnname"]().cast<std::string>();
        func =  builder.create<mllvm::LLVMFuncOp>(builder.getUnknownLoc(), name, typ);
        lower_func_body(func_ir);
        mod.push_back(func);
//        mod.dump();
        auto llvmmod = mlir::translateModuleToLLVMIR(mod);
//        llvmmod->dump();
        return py::bytes(serialize_mod(*llvmmod));
    }
private:
    mlir::MLIRContext ctx;
    mllvm::LLVMDialect& dialect;
    mlir::OpBuilder builder;
    mllvm::LLVMFuncOp func;
    mlir::Block::BlockArgListType fnargs;
    mlir::Block* entry_bb = nullptr;
    std::vector<mlir::Block*> blocks;
    std::vector<mlir::Value> locals;
    std::unordered_map<std::string, mlir::Value> vars;
    inst_handles insts;
    type_cache types;

    void lower_func_body(const py::object& func_ir)
    {
        entry_bb = func.addEntryBlock();
        assert(func.getNumArguments() >= 2);
        fnargs = func.getArguments().slice(2);
        auto ir_blocks = get_blocks(func_ir);
        assert(!ir_blocks.empty());
        blocks.resize(ir_blocks.size());
        std::generate(blocks.begin(), blocks.end(), [&](){ return func.addBlock(); });

        std::size_t i = 0;
        for (auto& it : ir_blocks)
        {
            lower_block(blocks[i], it.second);
            ++i;
        }

        builder.setInsertionPointToEnd(entry_bb);
        builder.create<mllvm::BrOp>(builder.getUnknownLoc(), mlir::None, blocks.front());
    }

    void lower_block(mlir::Block* bb, const py::handle& ir_block)
    {
        assert(nullptr != bb);
        builder.setInsertionPointToEnd(bb);
        for (auto it : get_body(ir_block))
        {
            lower_inst(it);
        }
    }

    void lower_inst(const py::handle& inst)
    {
        if (py::isinstance(inst, insts.Assign))
        {
            auto name = inst.attr("target").attr("name");
            auto val = lower_assign(inst, name);
            storevar(val, inst, name);
        }
        else if (py::isinstance(inst, insts.Del))
        {
            delvar(inst.attr("value"));
        }
        else if (py::isinstance(inst, insts.Return))
        {
            retvar(inst.attr("value").attr("name"));
        }
        else
        {
            report_error(llvm::Twine("lower_inst not handled: \"") + py::str(inst.get_type()).cast<std::string>() + "\"");
        }
    }

    mllvm::LLVMType get_ll_type(const py::handle& name)
    {
        return mllvm::LLVMType::getInt64Ty(&dialect); // TODO
    }

    mlir::Value resolve_op(mlir::Value lhs, mlir::Value rhs, const py::handle& op)
    {
        // TODO unhardcode
        if (op.is(insts.add))
        {
            return builder.create<mllvm::AddOp>(builder.getUnknownLoc(), lhs, rhs);
        }
        if (op.is(insts.eq))
        {
            assert(lhs.getType() == rhs.getType());
            if (lhs.getType().cast<mllvm::LLVMType>().isIntegerTy())
            {
                return builder.create<mllvm::ICmpOp>(builder.getUnknownLoc(), mllvm::ICmpPredicate::eq, lhs, rhs);
            }
        }
        if (op.is(insts.gt))
        {
            assert(lhs.getType() == rhs.getType());
            if (lhs.getType().cast<mllvm::LLVMType>().isIntegerTy())
            {
                return builder.create<mllvm::ICmpOp>(builder.getUnknownLoc(), mllvm::ICmpPredicate::sgt, lhs, rhs);
            }
        }

        report_error(llvm::Twine("resolve_op not handled: \"") + py::str(op).cast<std::string>() + "\"");
    }

    mlir::Value lower_binop(const py::handle& expr, const py::handle& op)
    {
        auto lhs_name = expr.attr("lhs").attr("name");
        auto rhs_name = expr.attr("rhs").attr("name");
        auto lhs = loadvar(lhs_name);
        auto rhs = loadvar(rhs_name);
        // TODO casts
        return resolve_op(lhs, rhs, op);
    }

    mlir::Value lower_expr(const py::handle& expr)
    {
        auto op = expr.attr("op").cast<std::string>();
        if (op == "binop")
        {
            return lower_binop(expr, expr.attr("fn"));
        }
        else if (op == "cast")
        {
            auto val = loadvar(expr.attr("value").attr("name"));
            // TODO cast
            return val;
        }
        else
        {
            report_error(llvm::Twine("lower_expr not handled: \"") + op + "\"");
        }
    }

    mlir::Value get_const_val(const py::handle& val)
    {
        if (py::isinstance<py::int_>(val))
        {
            auto mlir_type = mllvm::LLVMType::getIntNTy(&dialect, 64);
            auto value = builder.getI64IntegerAttr(val.cast<int64_t>());
            return builder.create<mllvm::ConstantOp>(builder.getUnknownLoc(), mlir_type, value);
        }
//        if (py::isinstance<bool>(val))
//        {
//            auto b = val.cast<int>();
//            auto mlir_type = mllvm::LLVMType::getInt1Ty(&dialect);
//            auto value = builder.getBoolAttr(b);
//            return builder.create<mllvm::ConstantOp>(builder.getUnknownLoc(), mlir_type, value);
//        }

        // assume it is a PyObject*
        auto mlir_type = mllvm::LLVMType::getInt8Ty(&dialect).getPointerTo();
        return builder.create<mllvm::NullOp>(builder.getUnknownLoc(), mlir_type);

//        report_error(llvm::Twine("get_const_val unhandled type \"") + py::str(val.get_type()).cast<std::string>() + "\"");
    }

    mlir::Value lower_assign(const py::handle& inst, const py::handle& name)
    {
        auto value = inst.attr("value");
        if (py::isinstance(value, insts.Arg))
        {
            auto index = value.attr("index").cast<std::size_t>();
            // TODO: incref
            // TODO: cast
            return fnargs[index];
        }
        if (py::isinstance(value, insts.Const) || py::isinstance(value, insts.Global))
        {
            // TODO unhardcode
            // TODO incref
//            auto mlir_type = mllvm::LLVMType::getIntNTy(&dialect, 64);
//            auto val = builder.getI64IntegerAttr(value.attr("value").cast<int64_t>());
//            return builder.create<mllvm::ConstantOp>(builder.getUnknownLoc(), mlir_type, val);
            return get_const_val(value.attr("value"));
        }
        if(py::isinstance(value, insts.Expr))
        {
            return lower_expr(value);
        }
        report_error(llvm::Twine("lower_assign not handled: \"") + py::str(value.get_type()).cast<std::string>() + "\"");
    }

    void alloca_var(const py::handle& name)
    {
        auto name_str = name.cast<std::string>();
        if (0 == vars.count(name_str))
        {
            scoped_goto_block s(builder, entry_bb);
            auto size_type =  mllvm::LLVMType::getIntNTy(&dialect, 64);
            auto size_val = builder.getI64IntegerAttr(/*TODO*/1);
            auto size = builder.create<mllvm::ConstantOp>(builder.getUnknownLoc(), size_type, size_val);
            auto type = get_ll_type(name);
            auto ptype = type.getPointerTo();
            auto op = builder.create<mllvm::AllocaOp>(builder.getUnknownLoc(), ptype, size, /*align*/0);
            auto null = zero_val(type);
            builder.create<mllvm::StoreOp>(builder.getUnknownLoc(), null, op);
            vars[name_str] = op;
        }
    }

    mlir::Value get_var(const py::handle& name)
    {
        auto it = vars.find(name.cast<std::string>());
        assert(vars.end() != it);
        return it->second;
    }

    mlir::Value loadvar(const py::handle& name)
    {
        auto type = get_ll_type(name);
        return builder.create<mllvm::LoadOp>(builder.getUnknownLoc(), type, get_var(name));
    }

    void storevar(mlir::Value val, const py::handle& inst, const py::handle& name)
    {
        alloca_var(name);
        auto old = loadvar(name);
        // TODO decref old
        auto ptr = get_var(name);
        builder.create<mllvm::StoreOp>(builder.getUnknownLoc(), val, ptr);
    }

    mlir::Value zero_val(mllvm::LLVMType type)
    {
        if (type.isPointerTy())
        {
            return builder.create<mllvm::NullOp>(builder.getUnknownLoc(), type);
        }
        else if (type.isIntegerTy())
        {
            return builder.create<mllvm::ConstantOp>(builder.getUnknownLoc(), type, builder.getI64IntegerAttr(0));
        }
        else
        {
            report_error(llvm::Twine("zero_val unhandled type ") + to_str(type));
        }
    }

    void delvar(const py::handle& name)
    {
        alloca_var(name);
        auto ptr = get_var(name);
        // TODO decref

        // TODO
        auto type = get_ll_type(name);
        auto null = zero_val(type);
        builder.create<mllvm::StoreOp>(builder.getUnknownLoc(), null, ptr);
    }

    void retvar(const py::handle& name)
    {
        alloca_var(name);
        auto val = loadvar(name);
        // TODO casts

        auto ret_ptr = func.getArgument(0);
        builder.create<mllvm::StoreOp>(builder.getUnknownLoc(), val, ret_ptr);

        auto mlir_type = mllvm::LLVMType::getIntNTy(&dialect, 32);
        mlir::Value ret = builder.create<mllvm::ConstantOp>(builder.getUnknownLoc(), mlir_type, builder.getI32IntegerAttr(0));
        builder.create<mllvm::ReturnOp>(builder.getUnknownLoc(), ret);
    }

    mllvm::LLVMType parse_type(llvm::StringRef str)
    {
        return types.get_type(dialect, str);
    }

    mllvm::LLVMType get_func_type(const py::handle& typedesc)
    {
        auto get_type = [&](const auto& h) {
            return parse_type(py::str(h).cast<std::string>());
        };
        auto p_func = typedesc();
        using Type = mllvm::LLVMType;
        auto ret = get_type(p_func.attr("return_type"));
        llvm::SmallVector<Type, 8> args;
        for (auto arg : p_func.attr("args"))
        {
            args.push_back(get_type(arg));
        }
        return Type::getFunctionTy(ret, args, false);
    }
};
}

py::bytes lower_function(const py::object& compilation_context, const py::object& func_ir)
{
    mlir::registerDialect<mllvm::LLVMDialect>();
    return lowerer().lower(compilation_context, func_ir);
}
