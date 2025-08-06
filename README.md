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
