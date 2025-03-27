from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class PerceptronClassifier:
    def __init__(self, random_state=None):
        """        
        :param max_iter: Nombre maximal d'itérations pour l'algorithme d'optimisation.
        :param tol: Tolérance pour l'arrêt.
        :param random_state: Graine pour le générateur de nombres aléatoires.
        :param penalty : régularisation du modèle, l1:régularisation Ridge, l2 : régularisation Lasso.
        :param eta0 : taux d'apprentissage initial
        """
        self.model = None  # Le modèle sera défini après GridSearchCV
        self.grid_params = {
            'max_iter': [2000, 3000, 5000],
            'tol': [1e-3, 1e-4, 1e-5],
            'penalty': [None, 'l2', 'l1'],  # Ajout de penalty
            'eta0': [0.01, 0.1, 1.0, 10.0]  # Ajout de eta0
        }
        self.random_state = random_state

    def fit(self, x, y):
        """
        Entraîne le perceptron sur les données d'entraînement x et les étiquettes y,
        en utilisant GridSearchCV pour trouver les meilleurs hyperparamètres.
        """

        base_model = Perceptron(random_state=self.random_state)

        # Détermination du nombre de fold possible
        _, occurrences = np.unique(y, return_counts=True)
        min_occurrence = np.min(occurrences)
        nb_fold = max(2, min(min_occurrence,5))

        # Configuration de GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.grid_params,
            cv=nb_fold,
            scoring='accuracy'
        )

        grid_search.fit(x, y)

        # Sauvegarde du meilleur modèle
        self.model = grid_search.best_estimator_ 
    
    def predict(self, x):
        """Prédit les classes pour les données x."""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")
        return self.model.predict(x)

    def fonction_de_decision(self, x):
        """Calcule les scores de décision pour les données x."""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")
        return self.model.decision_function(x)

    def probas_par_prediction(self, x):
        """
        Retourne les probabilités calculées à partir des scores de décision.
        Utilise la fonction sigmoid pour convertir les scores de décision en probabilités.
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")

        decision_scores = self.fonction_de_decision(x)

        if len(decision_scores.shape) == 1:  # Si binaire, transformer en 2D
            decision_scores = np.vstack([-decision_scores, decision_scores]).T

        # Appliquer la fonction sigmoid pour obtenir des probabilités
        probabilities = 1 / (1 + np.exp(-decision_scores))  # Sigmoid pour chaque score de décision

        return probabilities
