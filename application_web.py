# application_web.py

# ATTENTION! Ce serveur Flask expérimental ne doit pas être utilisé tel quel en production
# Utilisez plutôt un serveur WSGI

# Inspiration & droits d'auteur
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
# Copyright (c) 2018, Adrian Rosebrock, François Chollet
# https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker
# Copyright (c) 2018, Adrian Rosebrock, François Chollet
# https://www.analyticsvidhya.com/blog/2022/01/develop-and-deploy-image-classifier-using-flask-part-2/
# Copyright (c) 2022, Sajal Rastogi
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

