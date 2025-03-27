from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Pretraitement:
    def __init__(self, fichier_csv):
        self.fichier = fichier_csv
        self.x = None
        self.y = None
        self.classes = None
        self.labels_index = {}

    def mise_en_forme(self):
        """On supprime la colonne identifiant et on code les classes de manière numérique"""
        # On stocke les données et les étiquettes de classes dans des np.array
        x = np.loadtxt(
            self.fichier,
            delimiter=",",
            dtype=float,
            skiprows=1,
            usecols=range(2, 194)
        )
        labels_str = np.loadtxt(
            self.fichier,
            delimiter=",",
            dtype=str,
            skiprows=1,
            usecols=range(1, 2)
        )

        #On code les classes
        labels = np.array([], dtype=int)  # étiquette de classes pour chaque donnée
        idx = 0
        for label in labels_str:
            if label not in self.labels_index:
                self.labels_index[label] = idx
                idx += 1
            labels = np.append(labels, self.labels_index[label])

        self.classes = np.unique(labels)  # liste des classes

        self.x = x
        self.y = labels
        return x, labels

    def standardisation(self, x):
        """Standardisation des données"""
        return StandardScaler().fit_transform(x)
    
    def valeur_propre(self, x):
        """Affiche les valeurs propres de la matrice de covariance"""

        standardized_data = self.standardisation(x)
        covariance_matrix = np.cov(standardized_data.T)
        valeurs_propres = np.linalg.eig(covariance_matrix)[0]

        # Préparer les données pour le tableau
        indices = np.arange(len(valeurs_propres))
        data = list(zip(indices, valeurs_propres))

        # Créer une figure et une table
        fig, ax = plt.subplots(figsize=(8, 35))
        ax.axis('tight')
        ax.axis('off')

        # Ajouter le tableau
        table = ax.table(
            cellText=data,
            colLabels=["Indice", "Valeur Propre"],
            loc='center',
            cellLoc='center'
        )

        # Afficher
        plt.show()

        return valeurs_propres

    def pca(self, x, n_composantes, graph=False):
        """Retourne les données de dimension réduite après la transformation PCA à n_composantes"""
        # Transformation PCA
        standardized_data = self.standardisation(x)
        pca = PCA(n_components=n_composantes)
        reduced_x = pca.fit_transform(standardized_data)

        # Ajoute des étiquettes de classe
        reduced_data = np.column_stack((reduced_x, self.y))

        # Affichage d'un graphe optionnel
        if graph:
            if n_composantes == 2:
                self.graphique_2d_pca(reduced_data, pca.explained_variance_ratio_)
            elif n_composantes == 3:
                self.graphique_3d_pca(reduced_data, pca.explained_variance_ratio_)
            else:
                print("Le graphe associé au nombre de composantes demandé n'a pas été implémenté : essayez avec 2 ou 3")

        return reduced_data

    def graphique_2d_pca(self, np_data, var_explique):
        """Génère un nuage de points en deux dimensions avec n classes"""
        legend = ['Composante Principale 1', 'Composante Principale 2']

        # Récupération des classes
        classes = np_data[:, -1].astype(int)
        unique_classes = np.unique(classes)

        # Création du graphe
        plt.figure(figsize=(10, 7))
        for cls in unique_classes:
            indices = classes == cls
            plt.scatter(
                np_data[indices, 0],
                np_data[indices, 1],
                label=f'Classe {cls}',
                alpha=0.7
            )

        plt.title('Visualisation ACP des données', fontsize=16)
        var_explique_cp1 = int(var_explique[0] * 100)
        var_explique_cp2 = int(var_explique[1] * 100)
        plt.xlabel(
            f'{legend[0]} (Variation expliquée : {var_explique_cp1} %)',
            fontsize=12
        )
        plt.ylabel(
            f'{legend[1]} (Variation expliquée : {var_explique_cp2} %)',
            fontsize=12
        )
        plt.legend(
            title='Classe',
            loc='best',
            fontsize=8,
            ncol=4,
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.
        )
        plt.grid(True)
        plt.show()

    def graphique_3d_pca(self, np_data, var_explique):
        """Génère un nuage de points en trois dimensions"""

        legend = ['CP1', 'CP2', 'CP3']

        # Récupération des classes
        classes = np_data[:, -1].astype(int)
        unique_classes = np.unique(classes)

        # Création du graphe
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for cls in unique_classes:
            indices = classes == cls
            ax.scatter(
                np_data[indices, 0],
                np_data[indices, 1],
                np_data[indices, 2],
                label=f'Classe {cls}',
                alpha=0.7
            )

        plt.title('Visualisation ACP des données', fontsize=16)
        var_explique_cp1 = int(var_explique[0] * 100)
        var_explique_cp2 = int(var_explique[1] * 100)
        var_explique_cp3 = int(var_explique[2] * 100)
        ax.set_xlabel(
            f'{legend[0]} (Variation expliquée : {var_explique_cp1} %)',
            fontsize=8
        )
        ax.set_ylabel(
            f'{legend[1]} (Variation expliquée : {var_explique_cp2} %)',
            fontsize=8
        )
        ax.set_zlabel(
            f'{legend[2]} (Variation expliquée : {var_explique_cp3} %)',
            fontsize=8
        )
        plt.legend(
            title='Classe',
            loc='best',
            fontsize=6,
            ncol=3,
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.
        )
        plt.show()
