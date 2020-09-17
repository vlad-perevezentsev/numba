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

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include "plier/dialect.hpp"

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

template<typename T>
T& get_dialect(mlir::MLIRContext& ctx)
{
    auto dialect = ctx.getRegisteredDialect<T>();
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
        Branch = mod.attr("Branch");
        Jump = mod.attr("Jump");

        Arg = mod.attr("Arg");
        Expr = mod.attr("Expr");
        Var = mod.attr("Var");
        Const = mod.attr("Const");
        Global = mod.attr("Global");

        auto ops = py::module::import("operator");

        add = ops.attr("add");

        eq = ops.attr("eq");
        gt = ops.attr("gt");
    }

    py::handle Assign;
    py::handle Del;
    py::handle Return;
    py::handle Branch;
    py::handle Jump;

    py::handle Arg;
    py::handle Expr;
    py::handle Var;
    py::handle Const;
    py::handle Global;

    py::handle add;

    py::handle eq;
    py::handle gt;
};

struct type_cache
{
    using Type = mllvm::LLVMType;

    Type get_type(mlir::MLIRContext& context, llvm::StringRef str)
    {
        assert(!str.empty());
        auto s = str.str();
        auto it = typemap.find(s);
        if (typemap.end() != it)
        {
            return it->second;
        }
        auto type = parse_type(context, str);
        typemap[s] = type;
        return type;
    }

private:
    std::unordered_map<std::string, Type> typemap;
};

struct lowerer_base
{
    lowerer_base(): builder(&ctx) {}

protected:
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder;
    mlir::Block::BlockArgListType fnargs;
    std::vector<mlir::Block*> blocks;
    std::unordered_map<int, mlir::Block*> blocks_map;
    std::unordered_map<std::string, mlir::Value> vars;
    inst_handles insts;
};

struct lowerer : public lowerer_base
{
    lowerer():
        dialect(get_dialect<mllvm::LLVMDialect>(ctx))
    {

    }

    py::bytes lower(const py::object& compilation_context, const py::object& func_ir)
    {
        auto mod = mlir::ModuleOp::create(builder.getUnknownLoc());
        var_type_resolver = compilation_context["get_var_type"];
        auto typ = get_func_type(compilation_context["fntype"]);
        auto name = compilation_context["fnname"]().cast<std::string>();
        func =  builder.create<mllvm::LLVMFuncOp>(builder.getUnknownLoc(), name, typ);
        lower_func_body(func_ir);
        mod.push_back(func);
//        mod.dump();
        if (mlir::failed(mod.verify()))
        {
            report_error("MLIR module validation failed");
        }
        auto llvmmod = mlir::translateModuleToLLVMIR(mod);
//        llvmmod->dump();
        return py::bytes(serialize_mod(*llvmmod));
    }
private:
    mllvm::LLVMDialect& dialect;
    mllvm::LLVMFuncOp func;
    mlir::Block* entry_bb = nullptr;
    type_cache types;
    py::handle var_type_resolver;

    void lower_func_body(const py::object& func_ir)
    {
        entry_bb = func.addEntryBlock();
        assert(func.getNumArguments() >= 2);
        fnargs = func.getArguments().slice(2);
        auto ir_blocks = get_blocks(func_ir);
        assert(!ir_blocks.empty());
        blocks.reserve(ir_blocks.size());
        for (std::size_t i = 0; i < ir_blocks.size(); ++i)
        {
            blocks.push_back(func.addBlock());
            blocks_map[ir_blocks[i].first] = blocks.back();
        }

        for (std::size_t i = 0; i < ir_blocks.size(); ++i)
        {
            lower_block(blocks[i], ir_blocks[i].second);
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
        else if (py::isinstance(inst, insts.Branch))
        {
            branch(inst.attr("cond").attr("name"), inst.attr("truebr"), inst.attr("falsebr"));
        }
        else if (py::isinstance(inst, insts.Jump))
        {
            jump(inst.attr("target"));
        }
        else
        {
            report_error(llvm::Twine("lower_inst not handled: \"") + py::str(inst.get_type()).cast<std::string>() + "\"");
        }
    }

    mllvm::LLVMType get_ll_type(const py::handle& name)
    {
        return parse_type(py::str(var_type_resolver(name)).cast<std::string>());
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

    mlir::Value lower_call(const py::handle& expr)
    {
        auto args = expr.attr("args").cast<py::list>();
        auto vararg = expr.attr("vararg");
        auto kws = expr.attr("kws");
        // TODO fold args

        // TODO: hardcode for bool
        return loadvar(args[0].attr("name"));
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
        else if (op == "call")
        {
            return lower_call(expr);
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
        if(py::isinstance(value, insts.Expr))
        {
            return lower_expr(value);
        }
        if(py::isinstance(value, insts.Var))
        {
            auto var = loadvar(value.attr("name"));

            // TODO: cast
            // TODO: incref
            return var;
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

    void branch(const py::handle& cond, const py::handle& tr, const py::handle& fl)
    {
        auto c = loadvar(cond);
        auto tr_block = blocks_map.find(tr.cast<int>())->second;
        auto fl_block = blocks_map.find(fl.cast<int>())->second;
        // TODO: casts

        builder.create<mllvm::CondBrOp>(builder.getUnknownLoc(), c, tr_block, fl_block);
    }

    void jump(const py::handle& target)
    {
        auto block = blocks_map.find(target.cast<int>())->second;
        builder.create<mllvm::BrOp>(builder.getUnknownLoc(), mlir::None, block);
    }

    mllvm::LLVMType parse_type(llvm::StringRef str)
    {
        return types.get_type(ctx, str);
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

struct plier_lowerer : public lowerer_base
{
    plier_lowerer():
        dialect(get_dialect<plier::PlierDialect>(ctx))
    {

    }

    py::bytes lower(const py::object& compilation_context, const py::object& func_ir)
    {
        auto mod = mlir::ModuleOp::create(builder.getUnknownLoc());
        auto name = compilation_context["fnname"]().cast<std::string>();
        auto typ = get_func_type(compilation_context["fndesc"]);
        func = mlir::FuncOp::create(builder.getUnknownLoc(), name, typ);
        lower_func_body(func_ir);
        mod.push_back(func);
        mod.dump();
        if (mlir::failed(mod.verify()))
        {
            report_error("MLIR module validation failed");
        }
//        var_type_resolver = compilation_context["get_var_type"];
//        auto typ = get_func_type(compilation_context["fntype"]);
//        func =  builder.create<mllvm::LLVMFuncOp>(builder.getUnknownLoc(), name, typ);
//
//        auto llvmmod = mlir::translateModuleToLLVMIR(mod);
//        //        llvmmod->dump();
//        return py::bytes(serialize_mod(*llvmmod));
        return {};
    }
private:
    plier::PlierDialect& dialect;
    mlir::FuncOp func;

    void lower_func_body(const py::object& func_ir)
    {
        auto ir_blocks = get_blocks(func_ir);
        assert(!ir_blocks.empty());
        blocks.reserve(ir_blocks.size());
        for (std::size_t i = 0; i < ir_blocks.size(); ++i)
        {
            auto block = (0 == i ? func.addEntryBlock() : func.addBlock());
            blocks.push_back(block);
            blocks_map[ir_blocks[i].first] = block;
        }
        fnargs = func.getArguments();

        for (std::size_t i = 0; i < ir_blocks.size(); ++i)
        {
            lower_block(blocks[i], ir_blocks[i].second);
        }
    }

    void lower_block(mlir::Block* bb, const py::handle& ir_block)
    {
        assert(nullptr != bb);
//        vars.clear();
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
        else if (py::isinstance(inst, insts.Branch))
        {
            branch(inst.attr("cond").attr("name"), inst.attr("truebr"), inst.attr("falsebr"));
        }
        else if (py::isinstance(inst, insts.Jump))
        {
            jump(inst.attr("target"));
        }
        else
        {
            report_error(llvm::Twine("lower_inst not handled: \"") + py::str(inst.get_type()).cast<std::string>() + "\"");
        }
    }

    mlir::Value lower_assign(const py::handle& inst, const py::handle& name)
    {
        auto value = inst.attr("value");
        if (py::isinstance(value, insts.Arg))
        {
            auto index = value.attr("index").cast<std::size_t>();
            return builder.create<plier::ArgOp>(builder.getUnknownLoc(), index,
                                                name.cast<std::string>());
        }
        if(py::isinstance(value, insts.Expr))
        {
            return lower_expr(value);
        }
        if(py::isinstance(value, insts.Var))
        {
            auto var = loadvar(value.attr("name"));
            return builder.create<plier::AssignOp>(
                builder.getUnknownLoc(), var,
                value.attr("name").cast<std::string>());
        }
        if (py::isinstance(value, insts.Const))
        {
            auto val = get_const_val(value.attr("value"));
            return builder.create<plier::ConstOp>(builder.getUnknownLoc(), val);
        }
        if (py::isinstance(value, insts.Global))
        {
            auto name = value.attr("name").cast<std::string>();
            return builder.create<plier::GlobalOp>(builder.getUnknownLoc(),
                                                   name);
        }

        report_error(llvm::Twine("lower_assign not handled: \"") + py::str(value.get_type()).cast<std::string>() + "\"");
    }

    mlir::Value lower_expr(const py::handle& expr)
    {
        auto op = expr.attr("op").cast<std::string>();
        if (op == "binop")
        {
            return lower_binop(expr, expr.attr("fn"));
        }
        if (op == "cast")
        {
            auto val = loadvar(expr.attr("value").attr("name"));
            return builder.create<plier::CastOp>(builder.getUnknownLoc(), val);
        }
        if (op == "call")
        {
            return lower_call(expr);
        }
        report_error(llvm::Twine("lower_expr not handled: \"") + op + "\"");
    }

    mlir::Value lower_call(const py::handle& expr)
    {
        auto func = loadvar(expr.attr("func").attr("name"));
        auto args = expr.attr("args").cast<py::list>();
        auto kws = expr.attr("kws").cast<py::list>();
        auto vararg = expr.attr("vararg");
//        std::cout << py::str(args).cast<std::string>() << std::endl;
//        std::cout << py::str(kws).cast<std::string>() << std::endl;
//        std::cout << py::str(vararg).cast<std::string>() << std::endl;

        mlir::SmallVector<mlir::Value, 8> args_list;
        mlir::SmallVector<std::pair<std::string, mlir::Value>, 8> kwargs_list;
        for (auto a : args)
        {
            args_list.push_back(loadvar(a.attr("name")));
        }
        for (auto a : kws)
        {
            auto item = a.cast<py::tuple>();
            auto name = item[0];
            auto val_name = item[1].attr("name");
            kwargs_list.push_back({name.cast<std::string>(), loadvar(val_name)});
        }

        return builder.create<plier::PyCallOp>(builder.getUnknownLoc(), func,
                                               args_list, kwargs_list);
    }

    mlir::Value lower_binop(const py::handle& expr, const py::handle& op)
    {
        auto lhs_name = expr.attr("lhs").attr("name");
        auto rhs_name = expr.attr("rhs").attr("name");
        auto lhs = loadvar(lhs_name);
        auto rhs = loadvar(rhs_name);
        return resolve_op(lhs, rhs, op);
    }

    mlir::Value resolve_op(mlir::Value lhs, mlir::Value rhs, const py::handle& op)
    {
        // TODO unhardcode
        if (op.is(insts.add))
        {
            return builder.create<plier::BinOp>(builder.getUnknownLoc(), lhs, rhs, "+");
        }
//        if (op.is(insts.eq))
//        {
//            assert(lhs.getType() == rhs.getType());
//            if (lhs.getType().cast<mllvm::LLVMType>().isIntegerTy())
//            {
//                return builder.create<mllvm::ICmpOp>(builder.getUnknownLoc(), mllvm::ICmpPredicate::eq, lhs, rhs);
//            }
//        }
        if (op.is(insts.gt))
        {
            return builder.create<plier::BinOp>(builder.getUnknownLoc(), lhs, rhs, ">");
        }

        report_error(llvm::Twine("resolve_op not handled: \"") + py::str(op).cast<std::string>() + "\"");
    }

    void storevar(mlir::Value val, const py::handle& inst, const py::handle& name)
    {
        auto name_str = name.cast<std::string>();
        vars[name_str] = val;
    }

    mlir::Value loadvar(const py::handle& name)
    {
        auto it = vars.find(name.cast<std::string>());
        assert(vars.end() != it);
        return it->second;
    }

    void delvar(const py::handle& name)
    {
        auto var = loadvar(name);
        builder.create<plier::DelOp>(builder.getUnknownLoc(), var);
//        vars.erase(name.cast<std::string>());
    }

    void retvar(const py::handle& name)
    {
        auto var = loadvar(name);
        builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), var);
    }

    void branch(const py::handle& cond, const py::handle& tr, const py::handle& fl)
    {
        auto c = loadvar(cond);
        auto tr_block = blocks_map.find(tr.cast<int>())->second;
        auto fl_block = blocks_map.find(fl.cast<int>())->second;
        builder.create<mlir::CondBranchOp>(builder.getUnknownLoc(), c, tr_block, fl_block);
    }

    void jump(const py::handle& target)
    {
        auto block = blocks_map.find(target.cast<int>())->second;
        builder.create<mlir::BranchOp>(builder.getUnknownLoc(), mlir::None, block);
    }

    mlir::Attribute get_const_val(const py::handle& val)
    {
        if (py::isinstance<py::int_>(val))
        {
            return builder.getI64IntegerAttr(val.cast<int64_t>());
        }
        report_error(llvm::Twine("get_const_val unhandled type \"") + py::str(val.get_type()).cast<std::string>() + "\"");
    }

    mlir::FunctionType get_func_type(const py::handle& typedesc)
    {
        auto get_type = [&](const auto& h) {
//            return parse_type(py::str(h).cast<std::string>());
            return plier::PyType::get(&ctx);
        };
        auto p_func = typedesc();
        auto ret = get_type(p_func.attr("restype"));
        llvm::SmallVector<mlir::Type, 8> args;
//        for (auto arg : p_func.attr("argtypes"))
//        {
//            args.push_back(get_type(arg));
//        }
        return mlir::FunctionType::get(args, {ret}, &ctx);
    }
};
}

py::bytes lower_function(const py::object& compilation_context, const py::object& func_ir)
{
    mlir::registerDialect<mllvm::LLVMDialect>();
    plier::register_dialect();
    plier_lowerer().lower(compilation_context, func_ir);
    return lowerer().lower(compilation_context, func_ir);
}
