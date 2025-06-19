
# Handling Skewed Data in Regression

## 🔍 Understanding Skewness
- **Right (positive) skew**: Long tail on the right (e.g., income, house prices).
- **Left (negative) skew**: Long tail on the left.
- **Slight skew**: Might not significantly affect all models.

## 🤖 Model Sensitivity to Skewness
| Model Type        | Skew Sensitivity |
|-------------------|------------------|
| Linear Regression | High             |
| Tree-based Models (Random Forest, XGBoost) | Low |
| Neural Networks   | Medium           |
| kNN / SVM         | Medium to High   |

## 🛠️ Techniques to Handle Skewness

### 📏 Transforming the Target Variable
Only if the **target** is skewed:
- `log(y + ε)`
- Box-Cox or Yeo-Johnson

> Inverse-transform predictions for evaluation.

### 🧮 Transforming Features
- Use `log`, `sqrt`, `Box-Cox` or `Yeo-Johnson` for skewed features
- Normalize with `sklearn.preprocessing.FunctionTransformer` or pipelines

### 📊 Quantile / Binning Transformation
- Use `QuantileTransformer` or `PowerTransformer` to normalize distributions

## ✅ When to Transform Skewed Data

### 📌 Transform when:
1. Target is skewed in **linear models**.
2. Feature skew affects performance (e.g., in kNN or linear models).
3. Violations of model assumptions (normality, homoscedasticity).
4. Residuals show **non-constant variance**.
5. You want to **improve model performance** on long-tailed values.

### ❌ Don’t transform when:
- Using **tree-based models**.
- Skewness is **mild** (|skew| < 0.5).
- You prioritize **interpretability** over precision.

## 📏 How to Measure Skewness

### 📐 Formula (Pearson’s 3rd Skewness)
\`\`\`
from scipy.stats import skew
skew(df['your_column'])
\`\`\`

### 🔢 Interpretation
| Skewness | Interpretation       |
|----------|----------------------|
| ≈ 0      | Symmetric            |
| > 0      | Right-skewed         |
| < 0      | Left-skewed          |
| > 1 or < -1 | Highly skewed     |
| 0.5–1 or -0.5 to -1 | Moderately skewed |
| -0.5 to 0.5 | Approximately symmetric |

## 📊 Visual Methods

### Histogram
\`\`\`
df['your_column'].hist(bins=30)
\`\`\`

### Boxplot
\`\`\`
df.boxplot(column='your_column')
\`\`\`

### Q-Q Plot
\`\`\`
import scipy.stats as stats
stats.probplot(df['your_column'], dist="norm", plot=plt)
\`\`\`

---

> ✅ **Transform if the model performance or assumptions are affected.**  
> ❌ **Ignore skew for tree-based models or mild skewness.**



### 1  Why skewness matters (when you’re **not** using trees)

| Model family                                     | What goes wrong with a heavily-skewed variable                                                  | Symptoms you’ll see                                                      |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Linear / GLM (with identity link)**            | Coefficients chase extreme values, residuals show heavy tails, standard-error estimates inflate | Non-normal residual Q-Q plot, heteroscedastic fan shape                  |
| **Distance-based (k-NN, K-Means, SVM-RBF, PCA)** | One long tail can dominate Euclidean distance and principal components                          | Cluster assignment driven by one variable, PCs align with tail direction |
| **Gradient-based deep nets**                     | Large numeric range slows optimisation or explodes/vanishes gradients                           | Very small/large weights after first epochs                              |

Tree ensembles split by order, so they’re naturally robust—which is why we explicitly exclude them here.

---

### 2  Diagnosing “actionable” skew

1. **Plot** – histogram or KDE + a Q-Q plot.
2. **Quantify** – Pearson moment coefficient

$$
g_1=\frac{\sum_{i}(x_i-\bar x)^3/n}{s^3}
$$

| \|g₁\|  | Rule of thumb                                    |
| ------- | ------------------------------------------------ |
| < 0.5   | Essentially symmetric – no action needed         |
| 0.5 – 1 | Moderate skew – evaluate, may leave as is        |
| ≥ 1     | High skew – most models benefit from a transform |

3. **Model diagnostics** – fit a quick baseline model and inspect:

   * residual skew / kurtosis
   * residual-vs-fitted plot for heteroscedasticity
     If the problems disappear after a monotonic transform, keep it.

---

### 3  Transforming **features**

| Situation where transform **helps**            | Typical choice                   |
| ---------------------------------------------- | -------------------------------- |
| Positive, right-tailed counts or money amounts | `log1p(x)` or Box-Cox (λ≈0)      |
| Zero/negative allowed (e.g. deltas)            | Yeo-Johnson                      |
| Very long tail, outliers dominate distance     | Rank-based or quantile-to-normal |
| Variance grows with mean                       | Square-root or log               |

**Skip the transform** when:

* Skew is mild and coefficients must stay on the original scale for interpretability.
* The feature is categorical/ordinal represented as numbers.
* Down-stream business logic needs raw units (e.g., price elasticity modelling).
* You plan to feed the data to a model already robust to monotonic change later in the pipeline.

---

### 4  Transforming a **skewed target** (regression)

1. **Check residual behaviour first** – The target itself needn’t be normal; OLS only assumes *errors* ≈ normal and homoscedastic.
2. **Transform when**:

   * Errors fan out as $\hat y$ grows (variance non-constant).
   * Distribution has heavy right tail (g₁ ≥ 1) causing large-error penalties in RMSE.
   * Performance metrics in CV improve (MSE, $R^2$) after transformation.
3. **Choices**:

   * `log1p(y)` or Box-Cox for strictly positive $y$.
   * Yeo-Johnson if $y$ can be zero/negative.
4. **After fitting**:

   * Back-transform predictions: $\hat y = \exp(\hat z) - 1$ for log1p.
   * Re-evaluate on original scale with business-relevant metrics (MAE, MAPE).

**Don’t transform** when:

* Stakeholders consume predictions directly in natural units and the model already meets accuracy targets.
* You can instead adopt a distribution-aware model (e.g., Gamma GLM with log link, Zero-inflated Poisson) that natively handles skew.
* The primary loss function is robust to tails (e.g., quantile/Huber loss).

---

### 5  A practical decision workflow

```text
           ┌──► Fit quick baseline model
Data  ─Plot/Stats─► Residuals OK? ──► Yes ──► Keep as is
                │                    ▼
                │                 No
                │                    ▼
                └──► Try candidate transforms (log, √, Box-Cox, quantile)
                         │
                         ├─► Cross-validate; pick transform with
                         │    • best generalisation AND
                         │    • understandable coefficients
                         ▼
                    Finalise preprocessing pipeline
```

---

### 6  Key take-aways

* **Visual + statistical checks trump rules of thumb.** Always inspect residuals after a quick fit.
* **Monotonic power transforms** (log, Box-Cox, Yeo-Johnson) fix most numeric skew issues for linear and distance-based models.
* **Don’t over-transform.** If interpretability or business metrics suffer, a bit of skew is acceptable.
* For the target, consider a **distribution-specific model** (Gamma, Poisson, Tweedie) as an alternative to transforming.

Use these guidelines as guard-rails, but let cross-validated performance and diagnostic plots be the final judge.
