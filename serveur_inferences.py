# serveur_inferences.py
# ATTENTION! Ce serveur Flask expérimental ne doit pas être utilisé tel quel en production
# Utilisez plutôt un serveur WSGI

# Inspiration & droits d'auteur
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
# Copyright (c) 2018, Adrian Rosebrock, François Chollet
# https://www.analyticsvidhya.com/blog/2022/01/develop-and-deploy-image-classifier-using-flask-part-2/
# Copyright (c) 2022, Sajal Rastogi
# https://github.com:ClaudeCoulombe/EcorcIA_Web
# Copyright (c) 2022, Claude COULOMBE, adaptation, modification du code

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import requests
import json

# Données
HAUTEUR_IMAGE = 150
LARGEUR_IMAGE = 150

DICT_ARBRES = {
    'BOJ' : "Betula alleghaniensis - Bouleau jaune - Yellow birch",
    'BOP' : "Betula papyrifera - Bouleau à papier - White birch",
    'CHR' : "Quercus rubra - Chêne rouge - Northern red oak",
    'EPB' : "Picea glauca - Épinette blanche - White spruce",
    'EPN' : "Picea mariana - Épinette noire - Black spruce",
    'EPO' : "Picea abies - Épinette de Norvège - Norway spruce",
    'EPR' : "Picea rubens - Épinette rouge - Red spruce",
    'ERB' : "Acer platanoides - Érable de Norvège - Norway maple",
    'ERR' : "Acer rubrum - Érable rouge - Red maple",
    'ERS' : "Acer saccharum - Érable à sucre - Sugar maple",
    'FRA' : "Fraxinus americana - Frêne d'Amérique - White ash",
    'HEG' : "Fagus grandifolia - Hêtre à grandes feuilles - American beech",
    'MEL' : "Larix laricina - Mélèze - Tamarack",
    'ORA' : "Ulmus americana - Orme d'Amérique - American elm",
    'OSV' : "Ostrya virginiana - Ostryer de Virginie - American hophornbeam",
    'PEG' : "Populus grandidentata - Peuplier à grandes dents - Big-tooth aspen",
    'PET' : "Populus tremuloides - Peuplier faux tremble - Quaking aspen",
    'PIB' : "Pinus strobus - Pin blanc - Eastern white pine",
    'PID' : "Pinus rigida - Pin rigide - Pitch pine",
    'PIR' : "Pinus resinosa - Pin rouge - Red pine",
    'PRU' : "Tsuga canadensis - Pruche du Canada - Eastern Hemlock",
    'SAB' : "Abies balsamea - Sapin Baumier - Balsam fir",
    'THO' : "Thuja occidentalis - Thuya occidental - Northern white cedar",
}

CLASSES = list(DICT_ARBRES.keys())
ENTETE = {"charset":"utf-8", "content-type":"application/json"}
URL_MODELE = 'http://localhost:8501/v1/models/modele_1648351029:predict'

def pretraitement(image_entree):
    # Redimensionnement de l'image
    image_entree = image_entree.resize((HAUTEUR_IMAGE,LARGEUR_IMAGE))
    # Transformation en un tableau
    image_entree = img_to_array(image_entree)
    # Changement d'échelle de luminosité
    image_entree = image_entree / 255.
    # Créer un lot => (1, 150, 150, 3) pour le serveur
    image_entree = np.expand_dims(image_entree, axis=0)
    return image_entree

def appeler_serveur_docker(image_entree):
    # Transformer en données JSON
    image_entree_json = json.dumps({'instances': image_entree.tolist() })
    # Requête au serveur de modèle TensorFlow Serving qui tourne sur Docker
    reponse = requests.post(URL_MODELE, data=image_entree_json, headers=ENTETE)
    reponse_json = json.loads(reponse.text)
    # Extraire les prédictions du dictionnaire JSON retourné
    predictions = reponse_json['predictions']
    return predictions

def get_nomArbre(nom_classe):
    return DICT_ARBRES[nom_classe].split('-')[1]

# Code de prédiction / inférence
def predire(chemin_image):
    # Chargement et redimensionnement de l'image
    image_entree = load_img(chemin_image)
    image_entree = pretraitement(image_entree)
    predictions = appeler_serveur_docker(image_entree)
    # Obtenir la classe qui a la plus forte probabilité
    nom_classe = CLASSES[np.argmax(predictions[0])]
    nom_arbre = get_nomArbre(nom_classe)
    # Traces sur le serveur
    print(predictions[0])
    print(nom_classe,nom_arbre)
    return (nom_classe,nom_arbre)

# -------------------------------------------------------------------------------------------
# Licence
#
# Sous licence Apache License, Version 2.0 (la "Licence");
#
# Vous ne pouvez pas utiliser ce fichier sauf en conformité avec la licence.
#
# Vous pouvez obtenir une copie de la licence à: http://www.apache.org/licenses/LICENSE-2.0
#
# Sauf si requis par la loi applicable ou convenu par écrit, le logiciel
# distribué sous la Licence est distribué sur une BASE "TEL QUEL",
# SANS GARANTIE OU CONDITION D'AUCUNE SORTE, expresse ou implicite.
# Consultez la Licence pour connaître les autorisations et limitations
# spécifiques à la langue dans le cadre de la Licence.
# -------------------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
#
# You may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------------------------
