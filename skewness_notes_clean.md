
# Handling Skewed Features and Targets in Regression

## 1. Visualising & Quantifying Skewness

- **Plots**: Histogram (+ KDE), box‑plot, violin, empirical CDF, Q–Q plot.
- **Numeric measure**  
  ```python
  # unbiased (G1) skewness
  target_skew = df['target'].skew(bias=False)
  feature_skew = df['feature1'].skew(bias=False)
  ```
  `pandas.Series.skew(bias=False)` returns the bias‑corrected Fisher–Pearson coefficient $(G_{1}$).  
  For `scipy.stats.skew`, pass `bias=False` for the same estimator.

## 2. Practical Guidelines for Deciding on a Transform

| Absolute skew ($g_1$) | Typical action | Rationale |
|---------------------------|----------------|-----------|
| < 0.5 | Usually leave as‑is | Already close to symmetry; variance‑stabilising transforms rarely change cross‑validation score. |
| 0.5 – < 1.0 | Check residuals & CV | Try both raw and transformed; prefer the version that gives lower error **and** more homoscedastic residuals. |
| ≥ 1.0 | Transform is often helpful | Long right/left tail inflates squared‑error losses and heteroscedasticity; log/Box‑Cox/Yeo‑Johnson usually stabilise variance. |

Always verify with *empirical* cross‑validation rather than fixed thresholds.

### Choosing an evaluation metric

MSE / RMSE overweight extreme errors under heavy skew.  Prefer **MAE**, **Huber**, **quantile loss**, or compute RMSE after a *variance‑stabilising* transform (e.g. RMSLE when all targets > 0).

> **Caveat – RMSLE**  
> Works only for strictly positive targets and still penalises large *multiplicative* errors heavily. Not a general robust loss.

## 3. Model Sensitivity to Skewness

| Model family | Sensitivity to skew in **target** \(y\) | Notes |
|--------------|-----------------------------------------|-------|
| Ordinary least squares / Ridge / Lasso | High (if residuals heteroscedastic) | Linearity & equal‑variance assumptions apply to *residuals*. Skewed *predictors* are OK as long as residuals behave. |
| k‑NN, SVM (ε‑SVR) | Low – Medium | Sensitivity comes from chosen loss (squared, ε‑insensitive, MAE). |
| Neural nets | Medium | Sensitive if loss is squared‑error and no variance stabilisation; robust if using quantile or Huber loss. |
| Huber / quantile regression | Low | Built‑in robust loss. |
| Tree ensembles (RF, XGB, LGBM) | Low | Invariant to strictly **monotonic** transforms of features; default squared‑error loss on a *skewed target* still amplifies tail influence. Switch to Huber/quantile loss or transform *y*. |

## 4. Transforming the **Target** (\(y\))

### When to transform

- Residuals show heteroscedastic “megaphone” pattern.
- $(|g_{1}| ≥ 1)$ and CV error decreases after transform.
- Business metric cares about *relative* rather than absolute error.

### When **not** to transform

- Stakeholders need predictions in natural units and accuracy is acceptable.
- You already use a distribution‑aware model (e.g. Gamma GLM with log link).
- Robust loss (quantile/Huber) achieves desired accuracy.

### Common transforms

| Transform | Works with | `scikit‑learn` helper |
|-----------|------------|-----------------------|
| `log1p`   | positive data, right‑tail | `FunctionTransformer(np.log1p)` |
| **Box‑Cox** | positive data | `PowerTransformer(method="box-cox")` |
| **Yeo–Johnson** | positive & negative | `PowerTransformer(method="yeo-johnson")` |

**Tip:** wrap the whole thing in `TransformedTargetRegressor` to automate forward/backward transforms.  It removes boilerplate but does **not** pick the best λ; tune that with CV.

## 5. Transforming **Features**

### Transform when

- CV shows gain (often when $(|g_{1}| ≥ 1)$).
- The variance increases with the mean (classic log or square‑root fix).

### Avoid transforming when

- Interpretability of coefficients in raw units matters.
- Variable is categorical coded as an integer.
- A later model in the pipeline is rank‑invariant (tree, monotonic GBM).

### Useful transformers

| Purpose | Transformer |
|---------|-------------|
| Strict log / log1p | `FunctionTransformer(np.log1p)` |
| Box‑Cox / Yeo–Johnson | `PowerTransformer` |
| Map to normal / uniform by ranks | `QuantileTransformer` |

Use `ColumnTransformer` to apply transforms selectively.

## 6. Side Notes

<details>
<summary>Skewness Coefficients</summary>

*(Formulas identical to the original; omitted for brevity.)*
</details>

<details>
<summary>Homoscedasticity</summary>

- Homoscedastic: constant residual variance \( \operatorname{Var}(\epsilon_i)=\sigma^2 \).  
- Heteroscedastic: residual spread varies with \(\hat{y}\) or a predictor.  
- Diagnose with residual‑vs‑fitted plots, Breusch–Pagan, White, Goldfeld–Quandt tests.  
- Remedies: transform \(y\), robust/quantile loss, Weighted Least Squares, tree ensembles.
</details>

<details>
<summary>Predicting the extreme right tail</summary>

- Transform $(y)$ (`log` / `Box‑Cox`).  
- Use quantile loss (LightGBM, XGBoost) or Quantile Regression Forest.  
- Two‑stage "luxury vs regular" model if domain warrants different mechanisms.
</details>


