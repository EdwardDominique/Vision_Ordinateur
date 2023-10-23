# TP 1 : Pipeline complète de classification d'images

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score

np.random.seed(580)

X = 3 * np.random.rand(1000)

mean_X = round(np.mean(X), 2)
std_dev_X = round(np.std(X), 2)
median_X = round(np.median(X), 2)

print("La moyenne de X est:", mean_X)
print("l'écart type de X est:", std_dev_X)
print("La médiane de X est:", median_X)

X_bis = 3 * np.random.rand(1000)

mean_X_bis = round(np.mean(X_bis), 2)
std_dev_X_bis = round(np.std(X_bis), 2)
median_X_bis = round(np.median(X_bis), 2)

print("La moyenne de X_bis est:", mean_X_bis)
print("L'écart type de X_bis est:", std_dev_X_bis)
print("La médiane de X_bis est:", median_X_bis)

if mean_X == mean_X_bis:
    print("Les moyennes de X et X_bis sont égales.")
else:
    print("Les moyennes de X et X_bis sont différentes.")

if std_dev_X == std_dev_X_bis:
    print("Les écart types de X et X_bis sont égaux.")
else:
    print("Les écart types de X et X_bis sont différents.")

if median_X == median_X_bis:
    print("Les médianes de X et X_bis sont égales.")
else:
    print("Les médianes de X et X_bis sont différentes.")

noise = 0.1 * np.random.randn(1000)
y = np.sin(X) + noise

plt.figure(figsize=(8, 6))
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Visualisation de y en fonction de X')
plt.show()

plt.figure(figsize=(12, 4))
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Visualisation de y en fonction de X (Nouvelle taille)')
plt.show()

plt.hist(noise, bins=50, color='blue', alpha=0.7)
plt.xlabel('Valeur du bruit')
plt.ylabel('Fréquence')
plt.title('Histogramme du bruit gaussien')
plt.show()

# 2 - Données

dossier_bike = r'C:\Users\Utilisateur\Downloads\computer_vision_tp1\data1\bike'

images_bike = os.listdir(dossier_bike)
nombre_images_bike = len(images_bike)

print("Le nombre d'images dans le dossier bike est:", nombre_images_bike)

dossier_car = r'C:\Users\Utilisateur\Downloads\computer_vision_tp1\data1\car'

images_car = os.listdir(dossier_car)
nombre_images_car = len(images_car)

print("Le nombre d'images dans le dossier car est:", nombre_images_car)

def obtenir_info_image(chemin_image):
    img = Image.open(chemin_image)
    format_image = img.format
    taille_image = img.size
    return format_image, taille_image

print("Informations sur les images dans le dossier bike:")
for image in images_bike:
    chemin_image = os.path.join(dossier_bike, image)
    format_image, taille_image = obtenir_info_image(chemin_image)
    print(f"Nom de l'image: {image}")
    print(f"Format de l'image est: {format_image}")
    print(f"Taille de l'image est: {taille_image[0]} x {taille_image[1]} pixels")
    print()
    
print("Informations sur les images dans le dossier car :")
for image in images_car:
    chemin_image = os.path.join(dossier_car, image)
    format_image, taille_image = obtenir_info_image(chemin_image)
    print(f"Nom de l'image : {image}")
    print(f"Format de l'image est: {format_image}")
    print(f"Taille de l'image est: {taille_image[0]} x {taille_image[1]} pixels")
    print()
    
dossier_images = r'C:\Users\Utilisateur\Downloads\computer_vision_tp1\data1\bike'

fichiers_images = os.listdir(dossier_images)

nom_image = fichiers_images[86]

chemin_image = os.path.join(dossier_images, nom_image)

image = img.imread(chemin_image)

plt.imshow(image)
plt.title('Image en couleur')
plt.axis('off')
plt.show()

plt.imshow(image[:, :, 1], cmap="gray")
plt.title('Image en noir et blanc')
plt.axis('off')
plt.show()

image_inverse = np.flipud(image)

plt.imshow(image_inverse[:, :, 1], cmap="gray")
plt.title('Image à l\'envers')
plt.axis('off')
plt.show()

classes = ['bike', 'car']

target_size = (224, 224)

bike_folder = r'C:\Users\Utilisateur\Downloads\computer_vision_tp1\data1'
car_folder = r'C:/Users/Utilisateur/Downloads/computer_vision_tp1/data1'

def peuplate_images_and_labels_lists(image_folder_path):
    images = []
    labels = []

    for classe in classes:
        classe_folder = os.path.join(image_folder_path, classe)
        label = classes.index(classe)

        fichiers_images = os.listdir(classe_folder)

        for filename in fichiers_images:
            image_path = os.path.join(classe_folder, filename)
            image = cv2.imread(image_path)

            image = cv2.resize(image, target_size)

            images.append(image)
            labels.append(label)

    return images, labels

bike_images, bike_labels = peuplate_images_and_labels_lists(bike_folder)
car_images, car_labels = peuplate_images_and_labels_lists(car_folder)
    
images = np.array(bike_images + car_images)
labels = np.array(bike_labels + car_labels)

images = np.array([image.flatten() for image in images])
    
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

# 3 - Modèles de classification

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test[0:1])
print("La prédiction du label de la première image du set de test est:", prediction)

svm_clf = SVC(random_state=0)

svm_clf.fit(X_train, y_train)

svm_prediction = svm_clf.predict(X_test[0:1])
print("La prédiction du label avec le modèle de classification SVM est:", svm_prediction)

y_pred_model_1 = clf.predict(X_test)
accuracy_model_1 = accuracy_score(y_test, y_pred_model_1)
print("L'accuracy du modèle 1 est:", accuracy_model_1)

y_pred_model_2 = svm_clf.predict(X_test)
accuracy_model_2 = accuracy_score(y_test, y_pred_model_2)
print("L'accuracy du modèle 2 est:", accuracy_model_2)

confusion_matrix_model_1 = confusion_matrix(y_test, y_pred_model_1)
print("La matrice de confusion du modèle 1 est:", confusion_matrix_model_1)

confusion_matrix_model_2 = confusion_matrix(y_test, y_pred_model_2)
print("La matrice de confusion du modèle 2 est:", confusion_matrix_model_2)

precision_model_1 = precision_score(y_test, y_pred_model_1)
print("La précision du modèle 1 est:", precision_model_1)

recall_model_1 = recall_score(y_test, y_pred_model_1)
print("Le recall du modèle 1 est:", recall_model_1)

y_scores_model_1 = clf.predict_proba(X_test)[:, 1]
fpr_model_1, tpr_model_1, thresholds_model_1 = roc_curve(y_test, y_scores_model_1)
roc_auc_model_1 = roc_auc_score(y_test, y_scores_model_1)

plt.figure()
plt.plot(fpr_model_1, tpr_model_1, color='darkorange', lw=2, label=f'Modèle 1 (AUC = {roc_auc_model_1:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC du modèle 1')
plt.legend(loc="lower right")
plt.show()

# 4 - Comparaison de pipeline et fine tuning

tree_depth = clf.get_depth()
print("La profondeur de l'arbre de décision est:", tree_depth)

max_depth_list = list(range(1, 13))

train_accuracy = []
test_accuracy = []

for depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    train_accuracy.append(accuracy_score(y_train, y_train_pred))
    
    y_test_pred = clf.predict(X_test)
    test_accuracy.append(accuracy_score(y_test, y_test_pred))
    
plt.plot(max_depth_list, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_list, test_accuracy, label='Test Accuracy')

plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('L\'arbre de décision en fonction de l\'accuracy et max_depth')
plt.legend()
plt.show()

val_folder = r'C:\Users\Utilisateur\Downloads\computer_vision_tp1\val'

classes = ['bike', 'car']

target_size = (224, 224)

val_images = []
val_labels = []

for classe in classes:
    classe_folder = os.path.join(val_folder, classe)
    label = classes.index(classe)

    fichiers_images = os.listdir(classe_folder)

    for filename in fichiers_images:
        image_path = os.path.join(classe_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)

        val_images.append(image)
        val_labels.append(label)

val_images = np.array(val_images)
val_labels = np.array(val_labels)

meilleur_max_depth = 0.9727520435967303
meilleur_modele_1 = DecisionTreeClassifier(max_depth=meilleur_max_depth, random_state=0)
meilleur_modele_1.fit(X_train, y_train)

predictions_val = meilleur_modele_1.predict(val_images.reshape(len(val_images), -1))

accuracy_val = accuracy_score(val_labels, predictions_val)
print("L'accuracy des données de validation avec le modèle 1 est:", accuracy_val)