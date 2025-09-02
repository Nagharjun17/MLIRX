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
        auto zeros = SplatElementsAttr::get(tensorType, zeroAttr); //creates a tensor with 0.0s
        auto zeroConst = rewriter.create<toy::ConstantOp>(loc, zeros); //create a constant in toy dialect for further lowering

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