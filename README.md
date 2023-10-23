# TP 1 : Pipeline complète de classification d'images

Dans ce travail, nous devions explorer divers aspects de la vision par ordinateur en utilisant Python et des bibliothèques couramment utilisées telles que NumPy, scikit-learn (sklearn), et Matplotlib.

## Mise en Place de l’Environnement de Code

### Configuration du Répo Git

Pour commencer, j'ai configuré un dépôt Git pour ce projet afin de gérer le code de manière structurée.

### Code Interactif

J'ai utiliser un script Python via l'environnement de développement Spyder.

### Importation des Dépendances

J'ai installé et importé les bibliothèques nécessaires. Pour ce TP, j'utilise les bibliothèques suivantes :

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
```

### Manipulation des Bibliothèques

#### Numpy & Matplotlib
J'ai commencé par créer une liste X de 1000 points avec des valeurs aléatoires dans l'intervalle [0, 3] en utilisant np.random.rand.

Ensuite, j'ai calculé la moyenne, l'écart type et la médiane de cette liste avec les fonctions de NumPy.

Après cela, j'ai créé une autre liste X_bis de 1000 points avec des valeurs aléatoires dans le même intervalle et j'ai recalculé les statistiques.

J'ai également pris soin de fixer l'aléa en utilisant np.random.seed pour obtenir des résultats reproductibles.

J'ai généré une liste y de 1000 points en fonction de sin(X) à laquelle j'ai ajouté un bruit gaussien aléatoire d'amplitude de 10%.

Enfin, j'ai visualisé y en fonction de X sous forme de graphique scatter et j'ai créé un histogramme pour le bruit gaussien.

##### Questions : À quelle fonction la distribution de noise fait penser ? 
La distribution du bruit gaussien fait penser à une distribution normale ou gaussienne.

## Données
J'ai téléchargé les images du dossier data1 et effectué les étapes suivantes :

J'ai obtenu le format et la taille des images en utilisant la bibliothèque Pillow (PIL).

Ensuite, j'ai visualisé une image en couleur, en noir et blanc, et à l'envers pour illustrer ces concepts.

##### Questions : À quoi ressemble l'image en noir et blanc par rapport à l'image en couleur ? 
L'image en noir et blanc est une représentation en niveaux de gris de l'image en couleur.

#### Homogénéisation des Images
J'ai défini les chemins vers les dossiers "bike" et "car" pour les images, puis j'ai défini la taille souhaitée (target_size).

J'ai créé une méthode peuplate_images_and_labels_lists pour obtenir des images à la bonne dimension avec les labels correspondants.

Finalement, j'ai créé des arrays NumPy pour les images et les labels.

### Preprocessing des Images
J'ai représenté les images en créant une liste de taille (nb_image * nb_features).

### Séparation des Sets d'Entraînement et de Test
J'ai utilisé train_test_split pour diviser les images en ensembles d'entraînement et de test, avec 80% des images pour l'entraînement et 20% pour les tests.

## Modèles de Classification
Pour les modèles de classification, j'ai mis en place deux approches :

### Premier Modèle : Arbre de Décision
J'ai utilisé la classe DecisionTreeClassifier de sklearn.tree pour créer un modèle d'arbre de décision. J'ai ensuite entraîné le modèle et effectué des prédictions. J'ai calculé l'accuracy, généré une matrice de confusion et évalué la précision et le rappel du modèle.

### Deuxième Modèle : Support Vector Machine (SVM)
J'ai utilisé la classe SVC de sklearn.svm pour créer un modèle SVM, puis j'ai répété le processus d'entraînement et d'évaluation.

##### Questions : Comment prédire le label de la première image du set de test pour le modèle d'arbre de décision ? 
La prédiction du label de la première image du set de test pour le modèle d'arbre de décision peut être effectuée en utilisant la méthode predict(X_test[0:1]).

## Comparaison de Pipeline et Fine Tuning

Pour le modèle d'arbre de décision, j'ai exploré le fine-tuning en ajustant la profondeur de l'arbre (max_depth). J'ai créé un graphique pour visualiser comment l'accuracy varie en fonction de la profondeur.

J'ai utilisé également des données de validation pour trouver la meilleure valeur de max_depth et calculé l'accuracy correspondante.

##### Questions : Quelle est la meilleure valeur de max_depth à choisir ? Pourquoi ? 
La meilleure valeur de max_depth semble être environ 0.9727520435967303, car c'est à ce niveau de profondeur que le modèle atteint l'accuracy maximale avec les données de validation
