import numpy as np

class SansCaracteres:
    def __init__(self):
        self.classes = 2
        self.classe_predite = None

    def fit(self, x, y):
        """ 
        On prédut la classe la plus présente
        """
        if y is None or len(y) == 0:
            raise ValueError("Le vecteur y ne doit pas être vide.")

        count = np.bincount(y)
        self.classes = len(count)
        self.classe_predite = np.argmax(count)

    def predict(self, x):
        if self.classe_predite is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")
        
        return np.full(x.shape[0], self.classe_predite)

    def probas_par_prediction(self, x):
        """
        Renvoie la probabilité prédite pour chaque classe
        basée sur les caractéristiques en entrée x.
        """

        if self.classe_predite is None:
            raise ValueError("Le modèle doit être entraîné avec `fit` avant de faire des prédictions.")

        probabilites = np.zeros((x.shape[0], self.classes))
        probabilites[:, self.classe_predite] = 1.0

        return probabilites
