#ifndef PLIER_OPS
#define PLIER_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Plier_Dialect : Dialect {
  let name = "plier";
  let cppNamespace = "plier";
}

def Plier_Type : DialectType<Plier_Dialect,
                 CPred<"$_self.isa<::plier::Type>()">, "type">,
                 BuildableType<"$_builder.getType<::plier::Type>()"> {
}

class Plier_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Plier_Dialect, mnemonic, traits>;

def CastOp : Plier_Op<"cast", []> {
    let arguments = (ins
                         Plier_Type:$value);

    let builders = [
        OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value obj">
    ];
}


def DelOp : Plier_Op<"del", []> {
    let arguments = (ins
                       Plier_Type:$value);

    let builders = [
        OpBuilder<"OpBuilder &b, OperationState &state, ::mlir::Value obj">
    ];
}

#endif // PLIER_OPS
