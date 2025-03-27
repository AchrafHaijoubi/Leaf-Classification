import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class Evaluation:
    """Classe qui contient les méthodes pour évaluer les algorithmes de classification."""

    def __init__(self, x_test, y_test, algorithmes):
        """
        Initialise l'évaluation des données et des algorithmes.
        """
        self.x_test = x_test
        self.y_test = y_test
        self.algorithmes = algorithmes  #dictionnaire nom : classe_algo
        self.metrics_results = {
            nom: {"Precision": -1, "Recall": -1, "F1-Score": -1} for nom in algorithmes.keys()}

    def calculate_metrics(self):
        """
        Calcule les métriques (précision, rappel, F1-score).
        """
        for nom, algo in self.algorithmes.items():
            y_pred = algo.predict(self.x_test)
            precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
            self.metrics_results[nom] = {"Precision": precision, "Recall": recall, "F1-Score": f1}
        return self.metrics_results

    def plot_metrics(self):
        """Affiche les métriques sous forme de graphiques."""
        _ = self.calculate_metrics()

        # Mise des metrics sous forme de liste
        precision = []
        recall = []
        f1_score = []
        nom_algo = list(self.algorithmes.keys())

        for _, results in self.metrics_results.items():
            precision.append(results["Precision"])
            recall.append(results["Recall"])
            f1_score.append(results["F1-Score"])

        # Positionnement des barres
        x = np.arange(len(nom_algo))
        width = 0.25

        # Création des barres
        plt.bar(x - width, precision, width, label="Précision", color="#89CFF0")
        plt.bar(x, recall, width, label="Recall", color="#D8B7DD")
        plt.bar(x + width, f1_score, width, label="F1-Score", color="#F5F5DC")
        
        # Ajustement des étiquettes de l'axe x
        plt.xticks(rotation=45, ha='right')  # Rotation à 45 degrés, alignement à droite
        # Ajouter des détails au graphique
        plt.xlabel("Algorithmes")
        plt.ylabel("Score")
        plt.title("Comparaison des performances des algorithmes")
        plt.xticks(x, nom_algo)
        plt.ylim(0, 1.1)
        plt.legend()

        # Affichage
        plt.tight_layout()
        plt.show()

    def roc_par_methode(self, y_test, nom_algo_demande=[]):
        """
        Affiche un graphique des performances ROC des algorithmes choisis après entraînement
        """

        nom_algo = list(self.algorithmes.keys())
        nvlle_liste_nom = nom_algo.copy()
        # On considère les classes de test car sinon la courbe ne pourra pas être tracé
        # Il peut en effet ne pas avoir toutes les classes dans y_test
        # suite aux divisions de données
        classes_y_test = np.unique(y_test)
        classes_y_test = np.array([int(elt) for elt in classes_y_test])
        n_classes = len(classes_y_test)

        #Afin de tracer la courbe ROC on crée une matrice binaire, les lignes représentent les classes
        # elt = 1 <=> appartenance à la classe
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

        # Si on ne trace pas le graphique pour chaque méthode,
        # on doit mettre à jour la liste de nom des algos
        if nom_algo_demande !=[]:
            nvlle_liste_nom = []
            for elt in nom_algo_demande:
                if elt not in nom_algo:
                    print("Erreur : Vous n'avez pas initialisé la classe avec cette méthode;", elt)
                else:
                    nvlle_liste_nom.append(elt)

        pourcentages = {nom: [] for nom in nvlle_liste_nom}
        for nom in nvlle_liste_nom:
            algo = self.algorithmes[nom]
            y_scores = algo.probas_par_prediction(self.x_test)  # Scores de probabilité
            pourcentages[nom].append(y_scores)  # Sauvegarde des scores

            plt.figure(figsize=(10, 8))

            # Tracer les courbes ROC pour chaque classe
            fpr = {}  # Taux de faux positifs
            tpr = {}  # Taux de vrais positifs
            roc_auc = {}


            for c in classes_y_test:
                fpr[c], tpr[c], _ = roc_curve(y_test_bin[:, c], y_scores[:, c])
                roc_auc[c] = auc(fpr[c], tpr[c])

            for c in classes_y_test:
                plt.plot(fpr[c], tpr[c], label=f'{nom} - Classe {c} (AUC = {roc_auc[c]:.2f})')

            # Affichage des courbes
            plt.figure(figsize=(8, 6))
            for c in classes_y_test:
                plt.plot(fpr[c], tpr[c], label=f'Classe {c} (AUC = {roc_auc[c]:.2f})')

            plt.plot(fpr["micro"],
                     tpr["micro"],
                     linestyle='--',
                     color='gray',
                     label=f'ROC moyenne (micro-AUC = {roc_auc["micro"]:.2f})'
            )

            plt.title("Courbe ROC des Algorithmes de Classification")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel("Taux de Faux Positifs (FPR)")
            plt.ylabel("Taux de Vrais Positifs (TPR)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig("roc_auc_" + nom + ".png")
            plt.show()

    def roc_par_classe(self, y_test, numero_classe_demande=[]):
        """
        Affiche un graphique ROC par classe

        Affichage des performances ROC des
        algorithmes choisis après entraînement.
        Chaque graphe montre une courbe ROC par
        modèle pour une classe donnée.
        """
        nom_algo = list(self.algorithmes.keys())
        classes_y_test = np.unique(y_test)
        classes_y_test = np.array([int(elt) for elt in classes_y_test])
        nvlle_liste_classe = classes_y_test.copy()
        n_classes = len(classes_y_test)

        # Afin de tracer la courbe ROC, on crée une matrice binaire où les lignes représentent les classes
        # elt = 1 <=> appartenance à la classe
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

        # Si une liste de classes est demandée, on l'utilise
        if numero_classe_demande != []:
            nvlle_liste_classe = []
            for elt in numero_classe_demande:
                if elt not in classes_y_test:
                    print(f"Erreur : Classe {elt} non présente dans les données.")
                else:
                    nvlle_liste_classe.append(elt)

        pourcentages = {classe: [] for classe in nvlle_liste_classe}

        # Sauvegarde des scores pour chaque classe, pour chaque modèle
        for nom in nom_algo:
            algo = self.algorithmes[nom]
            y_scores = algo.probas_par_prediction(self.x_test)

            # Sauvegarde des scores pour chaque classe
            for classe in nvlle_liste_classe:
                pourcentages[classe].append(y_scores[:, classe])

        # Traçage des courbes ROC pour chaque modèle
        for c in nvlle_liste_classe:
            plt.figure(figsize=(10, 8))

            # Variables pour le tracé
            fpr = {}  # taux de faux positifs
            tpr = {}  # taux de vrais positifs
            roc_auc = {}

            # Paramètre courbes ROC pour chaque modèle
            for idx, nom in enumerate(nom_algo):
                fpr[nom], tpr[nom], _ = roc_curve(
                    y_test_bin[:, c],
                    pourcentages[c][idx]
                )
                roc_auc[nom] = auc(fpr[nom], tpr[nom])

                # Trace une courbe pour chaque modèle
                plt.plot(
                    fpr[nom],
                    tpr[nom],
                    label=f'{nom} - AUC = {roc_auc[nom]:.2f}'
                )

            # Paramétrage et affichage
            plt.title(f"Courbe ROC pour la classe {c}")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel("Taux de Faux Positifs (FPR)")
            plt.ylabel("Taux de Vrais Positifs (TPR)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(f"roc_auc_classe_{c}.png")
            plt.show()
