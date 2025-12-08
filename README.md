# **Functional Grid Search with Monads in Python**  
*A Functional, Side-Effect-Free, and Monadic Reimplementation of Scikit-Learn’s GridSearchCV.
It comprises evaluation improvements such as model selection based on the 1 Std.error rule (model-agnostic) and probability calibration (classification models) that improves performance in production and combats the feature-level bias.
*

---

## **1. Introduction**

Hyperparameter optimization is a central component of machine learning model development.  
While tools such as *Scikit-Learn’s* `GridSearchCV` provide robust functionality, they rely on:

- mutation and shared state,
- implicit exception-based error handling,
- object-oriented design tightly coupled to estimator interfaces,
- limited opportunities for functional composition and reasoning.

This project presents a **functional and monadic reimplementation** of Grid Search, inspired by:

- functional programming principles  
- explicit monadic error propagation  
- deterministic parallel execution  
- transparent model selection criteria  
- optional post-hoc **probability calibration** based on Expected Calibration Error (ECE)

The system produces reliable, reproducible results ideal for research environments, graduate-level education, and experimental ML system design.

---

## **2. Core Concepts**

### **2.1 Functional Design**

All computational components are written as **side-effect-free functions**:

- No mutation of global state  
- No silent exception propagation  
- Pure transformation pipelines  

This improves transparency, composability, and testability.

---

### **2.2 Monads for Error Handling**

Two monads form the backbone of the pipeline:

#### `Maybe[T]`
Represents optional values (Some or Nothing).  

#### `Result[T, E]`
Represents success (`Ok`) or failure (`Err`) as explicit computational states.

These abstractions eliminate the need for exception-driven control flow and promote safe composition of steps.

---

### **2.3 Validation Layer**

The validation subsystem ensures correctness before computation:

- estimator interface validation  
- feature / label array validation  
- parallelism configuration (`n_jobs`)  

Every validation result is wrapped in a `Result` monad.

---

### **2.4 Parallel Execution**

Parallel evaluation of model/parameter/fold combinations is implemented via:

- `ProcessPoolExecutor`  
- deterministic result aggregation  
- monadic wrapping of all task outcomes  

Parallelism is clean, safe, and compositional.

---

### **2.5 Model Selection with the One-Standard-Error Rule**

Beyond selecting the best parameter set, the system identifies:

> **The least complex model whose performance lies within one standard error of the best score.**

This 1-SE rule (commonly used in statistical learning such as CART and glmnet) favors parsimonious models and reduces overfitting tendencies.

---

### **2.6 (New) Probability Calibration for Classification Models**

For classification models supporting `predict_proba`, the system provides **optional probability calibration** using two established methods:

- **Isotonic Regression** (`method="isotonic"`)  
- **Platt Scaling** (`method="sigmoid"`)  

After refitting the best estimator on the entire dataset, the system:

1. Fits both calibration models.  
2. Computes the **Expected Calibration Error (ECE)** for each.  
3. Selects the calibration method with lower ECE.  
4. Stores:
   - `best_calibrated_estimator_`  
   - `calibration_results_` (ECE diagnostics)

This produces probabilistically reliable predictions crucial for:

- risk-sensitive applications  
- medical decision support  
- uncertainty modeling  
- calibrated inference pipelines  

Calibration is controlled via:

```python
calibrate=True