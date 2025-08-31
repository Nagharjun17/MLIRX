# MLIRX
This repository contains custom dialects, passes, and optimizations built on top of MLIR and LLVM for learning and exploring compiler engineering for machine learning.

### First, I clone the LLVM Project and set stuff up

Cloning into a directory called mlir-learning
```bash
git clone https://github.com/llvm/llvm-project.git
```

After building MLIR and LLVM from source, I configure, compile and test them finally storing the files inside a folder called build
```bash
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DCMAKE_BUILD_TYPE=Release
ninja check-mlir
```

I specifically built chapter 6 of the [Toy example](https://mlir.llvm.org/docs/Tutorials/Toy/) because it has the complete MLIR to LLVM lowering pipeline and is perfect for adding more features.
```bash
cd ~/mlir-learning/llvm-project/build
ninja toyc-ch6
```

Testing if the build is correct using a provided example by emitting LLVM code
```bash
./bin/toyc-ch6 ../mlir/test/Examples/Toy/Ch6/codegen.toy -emit=llvm
```

### Second, I add the operations I want to

Defining a new operation called squared relu
```tablegen
def SquareReLUOp : Toy_Op<"square_relu", [Pure, SameOperandsAndResultShape, ELementwise, NOperands<1>, NResults<1>]> {
  let summary = "squares element wise and applies relu activation";

  let argument = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}
```

For relu, I will also need an element wise max operation
```tablegen
def MaxOp : Toy_Op<"max", [Pure]> {
  let summary = "finds and returns elementwise max";

  let arguments = (ins AnyTensor:$input1, AnyTensor:$input2);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input1 `,` $input2 attr-dict `:` type($input1) `,` type($input2) `->` type($result)";

  let builders = [
    OpBuilder<(ins "Value":$input1, "Value":$input2)>
  ];
}
```

Making sure 
```cpp
#define GET_OP_CLASSES
#include "toy/Dialect.h.inc"
```
is present in `Dialect.h`

and

```cpp
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```
is present in `Dialect.cpp`

Above is because since we already built the chapter, we should have these definitions. Otherwise, we add them. These represent the linkage for the auto generated C++ code from the previous build from Ops.td


Now I go to
```bash
cd llvm-project/build
```



and build to generate C++ code for the new Square ReLU and Max operations
```bash
ninja toyc-ch6
```


Creating a lowering pass for Max operation inside `LowerToAffineLoops.cpp`
```cpp
#include "CustomLowering.h"

//,,,,,,,,,,,,,,,,,,,

struct MaxOpLowering : public OpConversionPattern<toy::MaxOp> {
  using OpConversionPattern<toy::MaxOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<toy::MaxOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(toy::MaxOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc(); //get location for future debugging
    
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      auto input1 = builder.create<affine::AffineLoadOp>(loc, adaptor.getInput1(), loopIvs); //load input 1
      auto input2 = builder.create<affine::AffineLoadOp>(loc, adaptor.getInput2(), loopIvs); //load input 2

      auto compare = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, input1, input2); //perform a comparison operation using a Ordered Greater Than predicate
      
      return builder.create<arith::SelectOp>(loc, compare, input1, input2); //apply the max operation and return
    });

    return success();
  }
};
```

Creating a lowering pass for Square ReLU operation in a separate file `CustomLowering.cpp` because it is much higher level than other ops.
```cpp
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "CustomLowering.h"

using namespace mlir;
using namespace toy;

namespace {
struct SquareReluLowering : public OpConversionPattern<SquareReLUOp> {
    using OpConversionPattern<SquareReLUOp>::OpConversionPattern;
    using OpAdaptor = typename OpConversionPattern<SquareReLUOp>::OpAdaptor;

    LogicalResult matchAndRewrite(SquareReLUOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        Value input = adaptor.getInput(); //get input value

        auto tensorType = llvm::cast<RankedTensorType>(input.getType()); //get input type and cast it to ranked tensor type
        auto elementType = llvm::cast<FloatType>(tensorType.getElementType()); //get element type inside tensor and cast it to float
        auto zeroAttr = rewriter.getFloatAttr(elementType, 0.0); //creating attribute representing constant 0.0 of type elementType
        auto zeros = DenseElementsAttr::get(tensorType, zeroAttr); //creates a tensor with 0.0s
        auto zeroConst = toy::ConstantOp::create(rewriter, loc, zeros); //create a constant in toy dialect for further lowering

        auto mul  = rewriter.create<toy::MulOp>(loc, input, input); //perform square operation
        auto relu = rewriter.create<toy::MaxOp>(loc, mul, zeroConst);

        rewriter.replaceOp(op, relu.getResult());
        return success();
    }
};
}

void mlir::toy::populateToyCustomLowerings(RewritePatternSet &patterns) {
  patterns.add<SquareReluLowering>(patterns.getContext());
}
```

Create new `CustomLowering.h` to register the newly created ops in other cpp files
```cpp
#pragma once
#include <mlir/IR/PatternMatch.h>

namespace mlir::toy {
  void populateToyCustomLowerings(mlir::RewritePatternSet &patterns);
}
```

Registering the patterns inside `LowerToAffineLoops.cpp`
```cpp
#include "CustomLowering.h"

//...........

patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering,
             MulOpLowering, PrintOpLowering, ReturnOpLowering,
             TransposeOpLowering,
             MaxOpLowering //created new MaxOp
             >
            (&getContext());
mlir::toy::populateToyCustomLowerings(patterns); //created new SquareReLUOp
```
Registering the ops in `Dialect.cpp`
```cpp
void MaxOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

void SquareReLUOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value input) {
  state.addTypes(input.getType());
  state.addOperands(input);
}
```

Emit specific operations for the newly defined ops inside `MLIRGen.cpp`.
```cpp
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    //..................

      return TransposeOp::create(builder, location, operands[0]);
    }

    if (callee == "square_relu") {
      if (operands.size() != 1) { emitError(location, "toy.square_relu needs 1 arg"); return nullptr; }
      return SquareReLUOp::create(builder, location, operands[0]);
    }
    
    if (callee == "max") {
      if (operands.size() != 2) { emitError(location, "toy.max needs 2 args"); return nullptr; }
      return MaxOp::create(builder, location, operands[0], operands[1]);
    }

    return GenericCallOp::create(builder, location, callee, operands);
  }
```


Since we created a new file `CustomLowering.cpp`, add `mlir/CustomLowering.cpp` to `mlir/examples/toy/Ch6/CMakeLists.txt` like given below:
```txt
add_toy_chapter(toyc-ch6
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  mlir/CustomLowering.cpp
```

Build again
```bash
ninja toyc-ch6
```
