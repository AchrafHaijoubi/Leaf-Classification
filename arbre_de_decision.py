from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class ArbreDeDecision:
    def __init__(self):
        self.model = None # Le modèle sera défini après GridSearchCV
    
    def fit(self, x, y):
        """ 
        Entraînement du modèle avec une validation croisée 
        pour trouver les meilleurs hyperparamètres 
        parmis une liste définie
        """
        model = DecisionTreeClassifier(random_state=42)

        #Paramètre que l'on va faire varier
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter':  ['best', 'random'],
            'max_depth': [None, 5, 70],
            'min_samples_split': [2, 4, 8],
            'max_features': [None, 'log2', 'sqrt' ],
        }

        ## Définition du nombre de fold possible
        # Etape 1 : on compte le nombre d'occurences par classes
        _, occurrences = np.unique(y, return_counts=True)
        # Etape 2 : on stocke le nombre d'occurences minimum
        min_occurrence = np.min(occurrences)
        # Etape 3 : On sélectionne le nombre de division
        nb_fold = max(2, min(min_occurrence,5))

        # On initialise la validation croisée
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=nb_fold,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        # On lance la validation croisée
        grid_search.fit(x, y)

        # Sauvegarde du meilleur modèle
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

