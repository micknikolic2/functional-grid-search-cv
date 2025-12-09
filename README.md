# **Functional Grid Search Cross-Validation**  
A functional, side-effect free, and monadic reimplementation of scikit-learn’s `GridSearchCV`.  
It comprises two major improvements:  
(1) model selection using the **one-standard-error rule**, and  
(2) optional **probability calibration** to reduce Expected Calibration Error (ECE) and mitigate feature-level bias.  

For detailed explanations of the one-standard-error rule and calibration, see Sections **2.5** and **2.6**.

---

## **1. Introduction**

Hyperparameter optimization is central to machine learning model development.  
While tools such as Scikit-Learn’s `GridSearchCV` provide powerful functionality, they rely on:

- mutation and shared state  
- exception-based control flow  
- tightly coupled object-oriented design  
- limited opportunities for functional composition  

This project presents a **functional and monadic reimplementation** of grid search, emphasizing:

- purity and referential transparency  
- explicit monadic error propagation  
- deterministic parallel execution  
- statistically grounded model selection  
- optional probability calibration  

The result is a reproducible, transparent, and mathematically principled system suitable for research, graduate coursework, and production-grade experimentation.

---

## **2. Core Concepts**

### **2.1 Functional Design**

All computational components are implemented as **side-effect-free functions**:

- no global state  
- no silent mutations  
- no hidden exception propagation  

This ensures:

- predictable behavior  
- better composability  
- straightforward unit testing  

---

### **2.2 Monadic Error Handling**

Two monads provide explicit control flow and safe computation:

#### **Maybe[T]**
Encodes optional values (`Some`, `Nothing`).

#### **Result[T, E]**
Encodes computational outcomes (`Ok(value)` or `Err(error)`).

These monads eliminate the need for exception-based logic and enable explicit, type-safe computation chains.

---

### **2.3 Validation Layer**

Every computational step begins with validation:

- estimator interface validation  
- feature/label array validation  
- cross-validation and parallelism configuration  

Results are always returned as `Result` monads, ensuring safe propagation of errors through the pipeline.

---

### **2.4 Parallel Execution**

Parallelization is implemented through:

- `ProcessPoolExecutor`  
- deterministic task assignment  
- monadic wrapping of all task outputs  

Parallelism is designed to be explicit, predictable, and reproducible.

---

### **2.5 Model Selection Using the One-Standard-Error Rule**

Rather than selecting only the parameter set with the highest mean cross-validation score, this system applies the **one-standard-error rule**:

> Select the *least complex* model whose mean CV score lies within one standard error of the best model.

This yields two estimators:

1. **`best_estimator_`** — highest mean performance  
2. **`one_se_estimator_`** — simpler model with comparable performance  

This principle, used in CART, glmnet, and classical statistical learning, produces models that generalize better and are less likely to overfit.

---

### **2.6 Probability Calibration for Classification Models**

If the estimator supports `predict_proba`, the system can optionally apply **post-hoc calibration** using:

- **Isotonic Regression**  
- **Platt Scaling (sigmoid)**  

Calibration is performed on cross-validated predictions of the refitted best estimator.

For each calibration method, Expected Calibration Error (ECE) is computed:

\[
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
\]

where:

- \( B_m \): calibration bin  
- \( N \): total samples  
- \( \text{acc}(B_m) \): empirical accuracy in bin  
- \( \text{conf}(B_m) \): mean predicted confidence in bin  

The method producing lower ECE is selected, and the system stores:

- `best_calibrated_estimator_`  
- `calibration_results_`  
- `one_se_calibrated_estimator_`  
- `one_se_calibration_results_`  

Calibration is activated with:

```python
calibrate=True


## **3. Summary of Outputs**

After calling:

```python
search.fit(X, y)

the system exposes several groups of outputs that capture estimator-level results and diagnostic information.

### **3.1 Estimator Outputs**

| Attribute                      | Description                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| `best_params_`                 | Parameter set achieving the best mean cross-validation performance                      |
| `best_estimator_`              | Estimator refitted on the entire dataset using the best parameter set                   |
| `one_se_estimator_`            | Estimator selected using the one-standard-error rule and refitted on the entire dataset |
| `best_calibrated_estimator_`   | Calibrated version of the best estimator (isotonic or sigmoid)                          |
| `one_se_calibrated_estimator_` | Calibrated version of the 1-SE estimator                                                |

### **3.2 Diagnostic Outputs**

| Attribute                     | Description                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------- |
| `best_score_`                 | Mean cross-validation score for the best model according to the refit metric                       |
| `best_index_`                 | Index of the best-performing parameter set in the parameter grid                                   |
| `cv_results_`                 | Comprehensive cross-validation results, including per-fold metrics, means, and standard deviations |
| `model_within_1se_`           | Summary of the selected 1-SE model, including its parameters and performance statistics            |
| `calibration_results_`        | Calibration diagnostics for the best estimator, including per-method ECE                           |
| `one_se_calibration_results_` | Calibration diagnostics for the 1-SE estimator                                                     |

Diagnostic include: mean and std of cross-validated scores, pre-fold training and validation metrics, comparison between competing models, calibration error measurements (ECE) across isotonic and sigmoid methods.

These outputs enable rigorous model comparison, transparency in model selection, and evaluation of probabilistic reliability.

