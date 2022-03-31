# application_web.py

# ATTENTION! Ce serveur Flask expérimental ne doit pas être utilisé tel quel en production
# Utilisez plutôt un serveur WSGI

# Inspiration & droits d'auteur
# Déployer un modèle d'apprentissage en profondeur avec Flask RESTful - EN FRANÇAIS
# https://thedatafrog.com/fr/articles/deploy-deep-learning-model-flask-restful/
# Copyright (c) 2021, Colin Bernet
# Building a simple Keras + deep learning REST API
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
# Copyright (c) 2018, Adrian Rosebrock, François Chollet
# Develop and Deploy Image Classifier using Flask: Part 2
# https://www.analyticsvidhya.com/blog/2022/01/develop-and-deploy-image-classifier-using-flask-part-2/
# Copyright (c) 2022, Sajal Rastogi
# EcorcIA Web, un serveur de modèles en ligne - avec TensorFlow Serving, Docker et Flask
# https://github.com:ClaudeCoulombe/EcorcIA_Web
# Copyright (c) 2022, Claude COULOMBE, adaptation, modification du code

import os
import serveur_inferences
from flask import Flask, request, url_for, render_template

app = Flask('serveur_inferences')

# Routage
@app.route('/', methods=['GET'])
def accueil():
    # Afficher la page d'accueil
    return render_template('accueil.html')

@app.route('/', methods=['POST'])
def inferer():
    fichier_televerse = request.files['fichier']
    if ( (fichier_televerse.filename != '') and
       (fichier_televerse.filename[-4:] in [".jpg","jpeg"]) ):
        chemin_image = os.path.join('static', fichier_televerse.filename)
        fichier_televerse.save(chemin_image)
        nom_classe, nom_arbre = serveur_inferences.predire(chemin_image)
        print("Nom de classe=",nom_classe)
        # Fournir un dictionnaire avec les informations à afficher
        dict_resultats = {
            'nom_classe':nom_classe,
            'nom_arbre':nom_arbre,
            'chemin_image':chemin_image,
        }
    else:
        # Problème avec le fichier (absent ou d'un mauvais type)
        dict_resultats = {
            'nom_classe':"*** ERREUR FICHIER ",
            'nom_arbre':" Pas de fichier photo en format .jpeg ou .jpg ***",
            'chemin_image':"/static/webpage_not_available-by_Lazur-openclipart_org.png",
            }
    return render_template('affichage.html',dict_resultats=dict_resultats)

if __name__ == '__main__':
    # Exécuter le serveur web sur le port 5000 du serveur local (localhost)
    app.run('localhost',5000,debug=True)

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
