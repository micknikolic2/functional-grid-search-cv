# Libraries

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from model_selection.functional_grid import FunctionalGridSearch

# Load dataset

def main():
        X,y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Define an estimator

        estimator = LogisticRegression(max_iter=2_000)

        # Grid with hyperparameter values 

        param_grid = {
            "C": [0.01, 0.1, 1.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        }

        # Functional GridSearchCV

        f2_scorer = make_scorer(fbeta_score, beta=2)
        acc_scorer = make_scorer(accuracy_score)

        search = FunctionalGridSearch(
            estimator=estimator,
            param_grid=param_grid,
            cv=5,
            scoring={'f2':f2_scorer, 'accuracy':acc_scorer},        
            n_jobs=-1,           
            return_train_score=True,
            refit='f2',
            calibrate=True,      
            verbose=-1
        ).fit(X,y)

        # Display results

        print("Best parameters:", search.best_params_)
        print("Best score (f2):", search.best_score_)
        print("Best calibrated estimator:", search.best_calibrated_estimator_)
        print("Calibration results:", search.calibration_results_)


if __name__ == '__main__':
        main()
