#1. Importez le jeu de données Digits.
from sklearn import datasets
digits = datasets.load_digits()

#2. Déterminez la dimension des données.
import numpy as np
data_dimension = np.shape(digits.data)
print(f"Dimension des données : {data_dimension}")
3. Affichez la description du jeu de données Digits.
print("Description du jeu de données Digits:")
print(digits.DESCR)
4. Affichez quelques images issues du jeu de données.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(2, 1))
for i, ax in enumerate(axes.ravel()):
ax.imshow(digits.images[i], cmap='gray_r')
ax.axis('off')
plt.show()

#5. Générer la matrice des liens à partir de la fonction linkage.
from scipy.cluster.hierarchy import linkage
matrice_liens = linkage(digits.data, method='ward')
6. Affichez le dendrogramme en utilisant la fonction dendrogram et en se basant sur la matrice
des liens. Déterminez le nombre optimal de cluster pour les chiffres manuscrits.
from scipy.cluster.hierarchy import dendrogram
dendrogram(matrice_liens, truncate_mode='level', p=3)
plt.title('Dendrogramme pour les chiffres manuscrits')
plt.xlabel('Échantillons')
plt.ylabel('Distance euclidienne')
plt.show()

#7. Découpez le dendrogramme en utilisant la fonction fcluster et identifiez les groupes de
chiffres dans chaque classe.
from scipy.cluster.hierarchy import fcluster
# Utilisez une valeur de seuil basée sur le dendrogramme pour déterminer le nombre de
clusters.
nombre_clusters_optimal = 10
clusters = fcluster(matrice_liens, nombre_clusters_optimal, criterion='maxclust')
# Afficher le nombre unique de clusters obtenus
nombre_clusters_obtenus = len(set(clusters))
print(f"Nombre de clusters obtenus : {nombre_clusters_obtenus}")
print(set(clusters))
clusters = clusters - 1
print(set(clusters))

#8. Affichez quelques images du jeu de données Digits avec les classes attribuées.
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
ax.imshow(digits.images[i], cmap='gray_r')
ax.set_title(f"Image {i+1}, Label: {digits.target[i]}, Cluster: {clusters[i]}")
ax.axis('off')
plt.show()

#10. A partir de la fonction cluster.KMeans, regroupez les chiffres manuscrits en 10 clusters
distincts.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_clusters = kmeans.fit_predict(digits.data)

#11. Présentez une visualisation des chiffres manuscrits avec leur catégories attribuées.
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
ax.imshow(digits.images[i], cmap='gray_r')
ax.set_title(f"Image {i+1}, KMeans Cluster: {kmeans_clusters[i]}")
ax.axis('off')
plt.show()

#12. Présentez un tableau de contingence des résultats issus des deux méthodes. Proposez une
interprétation en mettant en lumière les concordances ou divergences entre ces méthodes
de clustering.
from sklearn.metrics import confusion_matrix
import seaborn as sns
conf_matrix_hierarchy = confusion_matrix(digits.target, clusters)
conf_matrix_kmeans = confusion_matrix(digits.target, kmeans_clusters)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(conf_matrix_hierarchy, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Hierarchical Clustering")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
sns.heatmap(conf_matrix_kmeans, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title("KMeans Clustering")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
plt.show()
# (Insérez votre interprétation ici)
