Multicollinearity is primarily a concern in linear models, but it can still impact non‑linear models—just in different ways. Here's a breakdown:

🔹  In Linear Regression
   • What it does: If predictor variables are highly correlated, it becomes hard to determine the individual effect of each predictor.
   • Impact: Inflated standard errors, unstable coefficients, and poor interpretability.

🔹  In Non‑Linear Models (e.g., Decision Trees, Random Forests, Neural Nets)
   ✅ Less Sensitive
      Many non‑linear models don’t rely on coefficient estimates in the same way linear models do.
      Example: In decision trees or random forests, splits are made based on feature importance rather than solving equations.
      So multicollinearity won’t break the model or cause high variance in parameters.

   ⚠️ Still Has Drawbacks
      • Redundant features → Overfitting risk, especially in high‑capacity models like neural networks.
      • The model might learn spurious patterns, reducing generalization.
      • Training time & complexity: Extra correlated features increase dimensionality without adding new information, leading to longer training times and more complex models.
      • Feature importance can be misleading: Importance may be split across correlated features, making interpretation harder.
      • Interpretability tools can suffer: Tools like SHAP or LIME might give confusing results if multiple features carry the same signal.

💡  TL;DR
   • Linear regression: Multicollinearity directly affects the model’s reliability.
   • Non‑linear models: Multicollinearity doesn’t break the model but can lead to inefficiency, reduced interpretability, and potential overfitting.