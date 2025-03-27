import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

class ReseauxDeNeuronnes:
    def __init__(
            self,
            hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001,
            batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
            max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
            momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1,
            beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
    ):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                                    batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                                    power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol,
                                    verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                                    early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1,
                                    beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    def fit(self, x, y):
        param_grid = {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 50), (50, 100)],
            'activation': ['relu', 'logistic', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01, 0.1],
        }

        # Définition du nombre de fold possible
        _, occurrences = np.unique(y, return_counts=True)
        min_occurrence = np.min(occurrences)
        nb_fold = max(2, min(min_occurrence, 5))

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=nb_fold,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(x, y)

        # Sauvegarde du meilleur modèle
        best_model = grid_search.best_estimator_

        self.model = best_model

    def predict(self, x):
        """On prédit la bonne classe pour des données x"""
        return self.model.predict(x)

    def probas_par_prediction(self, x):
        """
        Renvoie la probabilité prédite pour chaque classe
        basée sur les caractéristiques en entrée x.
        """
        probas = self.model.predict_proba(x)

        print(probas)
        return probas
