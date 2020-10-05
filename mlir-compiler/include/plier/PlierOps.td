#ifndef PLIER_OPS
#define PLIER_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Plier_Dialect : Dialect {
  let name = "plier";
  let cppNamespace = "plier";
}

def Plier_PyType : DialectType<Plier_Dialect,
                 CPred<"$_self.isa<::plier::PyType>()">, "pytype">,
                 BuildableType<"$_builder.getType<::plier::PyType>()"> {
}

class Plier_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Plier_Dialect, mnemonic, traits>;

def ArgOp : Plier_Op<"arg", []> {
    let arguments = (ins
                     UI32Attr:$index,
                     StrAttr:$name);

    let results = (outs Plier_PyType);

    let builders = [
        OpBuilder<"OpBuilder &b, OperationState &state, unsigned index, StringRef name">
    ];
}

def ConstOp : Plier_Op<"const", []> {
    let arguments = (ins
                     AnyAttr:$val);

    let results = (outs Plier_PyType);

    let builders = [
        OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Attribute val">
    ];
}

def GlobalOp : Plier_Op<"global", []> {
  let arguments = (ins
                   StrAttr:$name);

  let results = (outs Plier_PyType);

  let builders = [
      OpBuilder<"OpBuilder &b, OperationState &state, StringRef name">
  ];
}

def BinOp : Plier_Op<"binop", []> {
  let arguments = (ins
                   Plier_PyType:$rhs,
                   Plier_PyType:$lhs,
                   StrAttr:$op);

  let results = (outs Plier_PyType);

  let builders = [
      OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value rhs, ::mlir::Value lhs, StringRef op">
  ];
}

def CastOp : Plier_Op<"cast", []> {
    let arguments = (ins
                     Plier_PyType:$value);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value val">
    ];
}

def PyCallOp : Plier_Op<"call", []> {
  let arguments = (ins
                   Plier_PyType:$func,
                   Variadic<AnyType>:$args,
                   UI32Attr:$kw_start,
                   ArrayAttr:$kw_names);

  let results = (outs Plier_PyType);

  let builders = [
      OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value func, "
                "::mlir::ValueRange args, "
                "::mlir::ArrayRef<std::pair<std::string, ::mlir::Value>> kwargs">
  ];
}

def BuildTupleOp : Plier_Op<"build_tuple", []> {
    let arguments = (ins
                     Variadic<AnyType>:$args);

    let results = (outs Plier_PyType);

    let builders = [
     OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::ValueRange args">
];
}

def StaticGetItemOp : Plier_Op<"static_getitem", []> {
    let arguments = (ins
                     Plier_PyType:$value,
                     Plier_PyType:$index_var,
                     UI32Attr:$index);

    let results = (outs Plier_PyType);

    let builders = [
     OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value value, "
               "::mlir::Value index_var, unsigned index">
];
}

def GetiterOp : Plier_Op<"getiter", []> {
    let arguments = (ins
                     Plier_PyType:$value);

    let results = (outs Plier_PyType);

    let builders = [
     OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value value">
];
}

def IternextOp : Plier_Op<"iternext", []> {
    let arguments = (ins
                     Plier_PyType:$value);

    let results = (outs Plier_PyType);

    let builders = [
     OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value value">
];
}

def PairfirstOp : Plier_Op<"pair_first", []> {
    let arguments = (ins
                     Plier_PyType:$value);

    let results = (outs Plier_PyType);

    let builders = [
     OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value value">
];
}

def PairsecondOp : Plier_Op<"pair_second", []> {
    let arguments = (ins
                     Plier_PyType:$value);

    let results = (outs Plier_PyType);

    let builders = [
     OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value value">
];
}

def DelOp : Plier_Op<"del", []> {
    let arguments = (ins
                     Plier_PyType:$value);

//    let builders = [
//        OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value val">
//    ];
}

#endif // PLIER_OPS
