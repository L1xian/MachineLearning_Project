#1. Importer le jeu de données.
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = load_iris()

#2. Déterminer les classes des fleurs du jeu de données.
classes_fleurs = iris.target_names
print("Classes de fleurs :", classes_fleurs)

#3. Afficher les features du jeu de données.
features = iris.feature_names
print("Features du jeu de données :", features)

#4. Affichez la description du jeu de données.
description = iris.DESCR
print("Description du jeu de données :\n", description)

#5. Déterminer la dimension du vecteur label de la base.
dimension_label = iris.target.shape
print("Dimension du vecteur label :", dimension_label)

#6. Déterminer la dimension de la matrice des features.
dimension_features = iris.data.shape
print("Dimension de la matrice des features :",
dimension_features)

#7. Afficher l’histogramme de la répartition des largeurs de pétales (petal width (cm)) dans
le jeu de données.
plt.figure(figsize=(8, 6))
plt.hist(iris.data[:, iris.feature_names.index("petal width
(cm)")], bins=30, edgecolor='k')
plt.xlabel("Largeur des pétales (petal_width)")
plt.ylabel("Fréquence")
plt.title("Histogramme de la répartition des largeurs de pétales")
plt.show()

#8. Afficher la distribution de probabilité des largeurs de pétales (petal width (cm)) du jeu de données.
plt.figure(figsize=(8, 6))
sns.kdeplot(iris.data[:, iris.feature_names.index("petal
width (cm)")], fill=True)
plt.xlabel("Largeur des pétales (petal_width)")
plt.ylabel("Distribution de probabilité")
plt.title("Distribution de probabilité des largeurs de pétales")
plt.show()

#9. Afficher la répartition du nombre d'exemples pour chaque espèce (species) en utilisant
un graphique.
plt.figure(figsize=(8, 6))
sns.countplot(x="species", data=pd.DataFrame(iris.target,
columns=["species"]))
plt.xlabel("Espèces")
plt.ylabel("Fréquence")
plt.title("Distribution de probabilité des espèces")
plt.show()

#10. Afficher la matrice de corrélation en utilisant la fonction heatmap.
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
correlation_matrix = df_iris.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",
square=True)
plt.title("Matrice de corrélation")
plt.show()

#Exercice 2 : Jeu de donnée TITANIC
#1. Importer le jeu de données.
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

#2. Sélectionner du jeu de données uniquement les caractéristiques suivantes : survived,
pclass, sex, age, sibsp, parch, fare, embarked, deck.
selected_features = ["survived", "pclass", "sex", "age",
"sibsp", "parch", "fare", "embarked", "deck"]
titanic = titanic[selected_features]
print(titanic.head())

#3. Traiter des données manquantes en supprimant les lignes qui contiennent un champ(NaN).
titanic = titanic.dropna()
print(titanic.head())

#4. Afficher la répartition du nombre d'exemples hommes et femmes en utilisant un graphique.
gender_counts = titanic["sex"].value_counts()
print("Nombre d'hommes et de femmes :\n", gender_counts)
sns.countplot(x='sex',data=titanic)
plt.title("Répartition des passagers par sexe")
plt.show()

#5. Afficher la répartition du nombre d'exemples hommes et femmes en fonction de la
variable pclass en utilisant un graphique.
sns.countplot(x='sex',hue='pclass',data=titanic)
plt.title("Répartition des passagers par classe et sexe")
plt.show()

#6. Affichez le nombre des femmes et des hommes en fonction de la variable survived (countplot).
sns.countplot(x='sex',hue='survived',data=titanic)
plt.title("Répartition des passagers par sexe et survie")
plt.show()

#7. Afficher la relation entre pclasse et age (catplot).
sns.catplot(x="pclass", y="age", data=titanic, kind="box")
plt.xlabel("Classe")
plt.ylabel("Âge")
plt.title("Relation entre la classe et l'âge")
plt.show()

#8. Afficher la relation entre pclasse, age et sex (catplot).
sns.catplot(x="pclass", y="age", hue="sex", data=titanic,
kind="box")
plt.xlabel("Classe")
plt.ylabel("Âge")
plt.title("Relation entre la classe, l'âge et le sexe")
plt.show()
