from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.special import softmax
import numpy as np
class SVMClassifier:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_accuracy = 0.0

    def fit(self, x, y):
        """
        Trouver les meilleurs hyperparamètres et entraîner le modèle.
        """
        # Initialisation le modèle SVM
        model = SVC()

        param_grid = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Types de noyaux
            'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2.0],  # Valeurs de régularisation
        }

        # Définition du nombre de fold minimum possible
        _, occurrences = np.unique(y, return_counts=True)
        min_occurrence = np.min(occurrences)
        nb_fold = max(2, min(min_occurrence, 5))

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=nb_fold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x, y)

        # Récupérer les meilleurs paramètres et modèle
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_accuracy = grid_search.best_score_


    def predict(self, x):
        """
        Prédire en utilisant le meilleur modèle entraîné.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")
        return self.model.predict(x)

    def fonction_de_decision(self, x):
        """
        Renvoie un score de décision pour chaque instance.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné.")

        # Utiliser la méthode decision_function pour obtenir les scores de décision
        decision_scores = self.model.decision_function(x)
        return decision_scores

    def probas_par_prediction(self, x):
        """
        Retourne les "probabilités" calculées à partir des scores de décision
        en utilisant une transformation softmax .
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")
        
        
        decision_scores = self.fonction_de_decision(x)

        if len(decision_scores.shape) == 1:  # Si binaire, transformer en 2D
            decision_scores = np.vstack([-decision_scores, decision_scores]).T

        # Utiliser la transformation softmax
        probabilities = softmax(decision_scores, axis=1)

        return probabilities