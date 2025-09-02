// #include "toy/Passes.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "toy/Dialect.h"

// using namespace mlir;

// namespace {
// struct ToyConstFoldPass : public PassWrapper<ToyConstFoldPass, OperationPass<ModuleOp>> {
//     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyConstFoldPass)
//     StringRef getArgument() const override { 
//         return "toy-fold-const"; 
//     }
//     StringRef getDescription() const override {
//         return "Run canonicalization patterns and per-op folders to a fixed point";
//     }

//     void runOnOperation() override {
//         RewritePatternSet patterns(&getContext());

//         // mlir::toy::AddOp::getCanonicalizationPatterns(patterns, &getContext());
//         // mlir::toy::MulOp::getCanonicalizationPatterns(patterns, &getContext());
//         // mlir::toy::CastOp::getCanonicalizationPatterns(patterns, &getContext());
//         // mlir::toy::TransposeOp::getCanonicalizationPatterns(patterns, &getContext());
//         // mlir::toy::ReshapeOp::getCanonicalizationPatterns(patterns, &getContext());
//         // mlir::toy::MaxOp::getCanonicalizationPatterns(patterns, &getContext());

//         GreedyRewriteConfig cfg;
//         cfg.setMaxIterations(GreedyRewriteConfig::kNoLimit);
//         (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), cfg);
//     }
// };
// }

// std::unique_ptr<mlir::Pass> mlir::toy::createConstFoldPass() {
//     return std::make_unique<ToyConstFoldPass>();
// }

// static mlir::PassRegistration<ToyConstFoldPass> pass;