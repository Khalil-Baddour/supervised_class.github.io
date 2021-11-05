# Import librairies
import sys
import os
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from osgeo import gdal
import numpy as np
import geopandas as gpd

# importer mes fonctions

sys.path.append('C:/chemein_du_dossier/fonctions')
import fonctions as fc

# 1 - Définition des inputs et outputs
# inputs
my_folder = 'C:/Users/chemin_dossier_des_donnes/'
my_sample = os.path.join(my_folder, 'img_sample.tif')
my_image = os.path.join(my_folder, 'image_a_classifier.tif')
#outputs
out_classif = os.path.join(my_folder, 'ma_classification.tif') # nommer mon resultat




# 2 --- Extraction des échantillons 
test_size = 0.5 # valeur à donner pour tester le modèle (entre 0 et 1)
X, Y, t = fc.get_samples_from_roi(my_image, my_sample)
# Fraction de mon échantillon en une portion pour entrainement et le test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

# 3 --- Entrainement du modèle

clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)
clf.fit(X_train, Y_train)

# 4 --- Tester le modèle
Y_predict = clf.predict(X_test)

# 5 --- Qualité du modèle
# Matrcice de confusion
cm = confusion_matrix(Y_test, Y_predict)
#Rapport de classification
report = classification_report(Y_test, Y_predict, labels=np.unique(Y_predict), output_dict=True)
# Accord global
accuracy = accuracy_score(Y_test, Y_predict)


# 6 --- Appliquer le modèle pour classifier l'image

# Charger l'image à classifier
X_img, _, t_img = fc.get_samples_from_roi(my_image, my_image)

# prédire l'image
Y_img_predict = clf.predict(X_img)

# Initialier mes résultas en tableau
ds = fc.open_image(my_image)
nb_col, nb_row, _ = fc.get_image_dimension(ds)

img = np.zeros((nb_row, nb_col, 1), dtype='uint8')
img[t_img[0], t_img[1], 0] = Y_img_predict

# Ecrire mon image et l'exporter
ds = fc.open_image(my_image)
fc.write_image(out_classif, img, data_set=ds, gdal_dtype=None,
            transform=None, projection=None, driver_name=None,
            nb_col=None, nb_ligne=None, nb_band=1)
