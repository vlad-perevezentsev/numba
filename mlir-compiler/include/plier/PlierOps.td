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

def CastOp : Plier_Op<"cast", []> {
    let arguments = (ins
                     Plier_PyType:$value);

    let results = (outs Plier_PyType);

    let builders = [
        OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value val">
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
