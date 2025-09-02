#pragma once
#include <mlir/IR/PatternMatch.h>

namespace mlir::toy {
  void populateToyCustomLowerings(mlir::RewritePatternSet &patterns);
}