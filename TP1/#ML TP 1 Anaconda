#1. Créer une liste T contenant les entiers de 1 à 10.
T= list(range(1, 11))
print('T=',T)

#2. Convertir la liste T en un tableau NumPy.
import numpy as np
T=np.array(T)
print('T=',T)

#3. Convertir le type des éléments de T en réel, puis afficher T.
T=T.astype('float')
print('T=',T)

#4. Afficher le nombre des éléments de T.
print('Le nombre des éléments T',T.size)

#5. Afficher le type des éléments de T.
print('Le type des éléments du tableau',T.dtype)

#6. Réorganiser le tableau T pour qu'il devienne une matrice de dimensions (2,5), puis
sauvegarder cette nouvelle disposition dans une variable appelée T2 et afficher T2.
T2=T.reshape(2,5)
print('T2',T2)

#7. Affichez la valeur minimale, maximale et la moyenne des éléments de T.
print('La valeur minimale de T', T.min())
print('La valeur maximale de T', T.max())
print('La valeur de la moyenne des éléments de T', T.mean())

#8. Afficher la somme des éléments de T.
print('La somme des éléments de T',T.sum())

#9. Afficher la valeur du premier et du dernier élément de T.
print('La valeur du premier et du dernier élément de
T',T[[0,9]])
      
#10. Afficher la valeur des éléments d’indice impair. 
print('La valeur des éléments d''indice impair',T[1::2])
      
#11. Insérer la valeur 20 la deuxième position et l’élément 11 à la fin du tableau, puis 
afficher T.
T=np.insert(T,1,20)
print(T)
T=np.append(T,11)
print(T)

#12. Supprimer les quatre premiers éléments de T et l’afficher.
T=np.delete(T,range(4))
print(T)

#13. Diviser T en deux tableaux de même dimension et les sauvegarder dans des variables
Tl et Tr.
TL,TR=np.hsplit(T,2)
print(TL,TR)

#14. Concaténer les Tl et Tr en une matrice M et afficher M.
M=np.vstack((TL,TR))
print(M)
M=np.hstack((TL,TR))
print(M)
