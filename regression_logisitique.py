from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

class RegressionLogistique:
    def __init__(self):
        self.model = None # Le modèle sera défini après GridSearchCV

    def fit(self, x, y):
        """ 
        Entraînement du modèle avec une validation croisée 
        pour trouver les meilleurs hyperparamètres 
        parmis une liste définie
        """
        model = LogisticRegression()

        # Configuration de GridSearchCV
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'max_iter' :  [1000, 1500, 3000, 5000],
            'solver': ['saga', 'newton-cg', 'sag'],
            'fit_intercept' : [True, False],
        }

        # Définition du nombre de fold possible
        _, occurrences = np.unique(y, return_counts=True)
        min_occurrence = max(2, np.min(occurrences))
        nb_fold = min(min_occurrence, 5)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=nb_fold,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(x, y)

        # Utilisation des meilleurs hyperparamètres
        best_model = grid_search.best_estimator_
        self.model = best_model

    def predict(self, x):
        """On prédit la bonne classe pour des données x"""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")
        y_pred = self.model.predict(x)
        return y_pred

    def probas_par_prediction(self, x):
        """
        Renvoie la probabilité prédite pour chaque classe
        basée sur les caractéristiques en entrée x.
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")
        
        proba_pred = self.model.predict_proba(x)
        return proba_pred
