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

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \, \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|
$$

where:

- \(B_m\): calibration bin  
- \(N\): total samples  
- \acc_(B_m)\): empirical accuracy in bin m  
- \conf_(B_m)\): mean predicted confidence in bin m 

The method producing lower ECE is selected, and the system stores:

- `best_calibrated_estimator_`  
- `calibration_results_`  
- `one_se_calibrated_estimator_`  
- `one_se_calibration_results_`  

Calibration is activated with:

`calibrate=True`


## 3. Summary of Outputs

After calling:

search.fit(X, y)

the system exposes two categories of outputs:
(1) estimator-level results and  
(2) diagnostic structures describing model performance, calibration, and cross-validation behaviour.

---

### 3.1 Estimator Outputs

best_params_  
The hyperparameter combination achieving the best mean cross-validation score.

best_estimator_  
The estimator refitted on the full dataset using best_params_.

one_se_estimator_  
The estimator chosen under the one-standard-error rule and refitted on the full dataset.

best_calibrated_estimator_  
The calibrated version of the best estimator (isotonic or sigmoid), selected by lowest Expected Calibration Error (ECE).

one_se_calibrated_estimator_  
The calibrated version of the one-standard-error estimator.

---

### 3.2 Diagnostic Outputs

best_score_  
The highest mean cross-validation score according to the refit metric.

best_index_  
The index of the best-performing parameter configuration.

cv_results_  
A comprehensive structure containing per-fold metrics, mean and standard deviation of scores, timing information, and parameter-level outputs.  
The layout mirrors scikit-learn's GridSearchCV.

model_within_1se_  
A dictionary describing the estimator selected under the one-standard-error rule.

calibration_results_  
ECE diagnostics for the best estimator, comparing isotonic and sigmoid calibration.

one_se_calibration_results_  
ECE diagnostics for the one-standard-error estimator.

---

### Diagnostics Include

- per-fold training and validation metrics  
- mean and standard deviation of cross-validation scores  
- fit and score time statistics  
- ranking of models under the selected metric  
- calibration error comparisons (ECE)  
- complete information for reproducible model selection

These outputs support transparent, rigorous evaluation suitable for research, teaching, and production ML pipelines.

## 4. Implementation Overview

The system is organized into modular components, each responsible for a single aspect of the grid search procedure. The implementation avoids mutation, relies on pure functions, and expresses all processing stages through explicit data transformations.

### 4.1 Parameter Grid Expansion

Parameter grids are expanded into explicit combinations using a deterministic product of values. Validation ensures that all entries follow scikit-learn conventions. The result is a sequence of dictionaries representing individual hyperparameter settings.

### 4.2 Cross-Validation Strategy

The fitting routine chooses between KFold and StratifiedKFold depending on whether the estimator reports a classifier type. For each parameter combination and each fold, a task is created containing all required computation inputs. These tasks form the basis for parallel evaluation.

### 4.3 Parallel Evaluation

Evaluation occurs through a functional wrapper over a process-based executor.  
Key characteristics:

- each task is pure and free of side effects  
- inputs and outputs are serialized without shared state  
- results propagate through monadic containers  
- task failures are safely captured and reported  

Parallel execution produces a collection of Result objects, which are filtered and aggregated.

### 4.4 Metric Handling

The scoring subsystem supports both:

- single-metric evaluation  
- multi-metric evaluation via a mapping of names to scorer functions  

Cross-validation results record:

- per-fold test metrics  
- optional training metrics  
- mean and standard deviation for each metric  
- ranking of candidates  

The structure mirrors scikit-learn’s cv_results_ conventions.

### 4.5 Model Selection

Two selection criteria are implemented:

1. standard best-mean-score selection  
2. the one-standard-error rule for selecting a simpler model with performance statistically indistinguishable from the best  

This rule improves generalization and stabilizes the choice of hyperparameters in small or noisy datasets.

### 4.6 Probability Calibration

For estimators supporting predict_proba, optional probability calibration may be applied.  
Calibration evaluates:

- isotonic regression  
- sigmoid (Platt scaling)  

Expected Calibration Error (ECE) is computed on held-out folds, and the method with the lower ECE is retained. Diagnostics are stored for further inspection.

---

## 5. Example Usage

The following illustrates a typical workflow:

1. define an estimator  
2. specify a parameter grid  
3. define scoring  
4. run the functional grid search  
5. inspect both estimator and diagnostic outputs  

The system can be used as a drop-in conceptual replacement for GridSearchCV while remaining transparent and suitable for research and experimentation.

---

## 6. Design Principles

Several principles guided the construction of the framework:

### 6.1 Functional Programming

All major operations are expressed as pure functions whose outputs depend solely on inputs. This improves composability and supports clear reasoning about the behaviour of the system.

### 6.2 Explicit Error Handling

Validation failures, numerical issues, and task-level errors are wrapped in Result monads. This pattern avoids exception-driven control flow and ensures that failure modes remain explicit and inspectable.

### 6.3 Deterministic Behaviour

Parameter expansion, fold generation, scoring, and aggregation follow deterministic ordering. This ensures reproducibility and simplifies debugging.

### 6.4 Transparency

Unlike traditional black-box grid search, every component of the process is visible. Intermediate states may be interrogated, extended, or replaced for research-oriented modifications.

---

## 7. Applications

This framework is intended for:

- graduate-level machine learning coursework  
- research experiments where transparency and control are necessary  
- comparative studies of estimators and calibration methods  
- environments requiring deterministic and interpretable model selection  
- methodological investigations such as the study of bias, variance, calibration, or model complexity  

The design also facilitates integration with:

- probabilistic pipelines  
- uncertainty-aware systems  
- custom model selection rules  
- functional data science workflows  

---

If integrated into production systems, calibrated outputs and conservative model selection criteria help mitigate overconfidence, reduce rare-event misclassification risk, and strengthen the reliability of downstream components.