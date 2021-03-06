# <b>EcorcIA Web</b>, un serveur d'inférences en ligne

## Avec TensorFlow Serving, Docker et Flask

## Introduction

Cours VIARENA - applications de la VIsion ARtificielle dans les REssources NAturelles.

Dans ce laboratoire vous allez créer un service d'inférence (ou serveur de modèles) capable d'identifier un arbre à partir d'une photo de son écorce téléversée sur le serveur. Vous allez donc déployer un modèle précédemment entraîné que vous avez téléchargé et sauvegardé sur votre poste de travail.</p>

<p>Vous utiliserez <code>TensorFlow Serving</code>, un serveur de modèles en source libre qui offre des services performants pour les environnements de production.</p>

<p>Il existe plusieurs façons d'installer TensorFlow Serving. La façon la plus simple est l'exécution dans un conteneur <a href="https://www.docker.com" target="_blank">Docker</a>.

## Prérequis

Connaissance du développement d'une applicaion Web

<h3><b>Inspiration et droits d'auteur</b></h3>

<p>Ce laboratoire s'inspire de plusieurs oeuvres publiques ou en logiciels libres qui ont été transformées dont:</p>
<ul>
  <ul>
    <li><a href="https://thedatafrog.com/fr/articles/deploy-deep-learning-model-flask-restful/" target='_blank'>Déployer un modèle d'apprentissage en profondeur avec Flask RESTful</a> - EN FRANÇAIS<br/>
    Copyright (c) 2021, Colin Bernet</li>
  <li><a href="https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html" target='_blank'>Building a simple Keras + deep learning REST API</a><br/>
  Copyright (c) 2018, Adrian Rosebrock, François Chollet</li>
  <li><a href="https://www.analyticsvidhya.com/blog/2022/01/develop-and-deploy-image-classifier-using-flask-part-2/" target='_blank'>Develop and Deploy Image Classifier using Flask: Part 2</a><br/>
  Copyright (c) 2022, Sajal Rastogi</li>
  <li><a href="https://github.com:ClaudeCoulombe/EcorcIA_Web" target='_blank'>EcorcIA Web, un serveur d'inférences en ligne</a> - avec TensorFlow Serving, Docker et Flask<br/>
  Copyright (c) 2022, Claude COULOMBE, adaptation, modification du code</li>
  </ul>
</ul>

## Support

Avertissement! Ce serveur Flask expérimental ne doit pas être utilisé tel quel en production. 
Utilisez plutôt un serveur WSGI Python de capacité industrielle ou encore NginX, Node.js ou Apache.

Cours VIARENA: Vous devez comprendre que nous ne pourrons offrir aucun support technique
en dehors du support communautaire du forum de ce cours.

## Données

Les données sur les écorces d'arbres proviennent de <a href="https://data.mendeley.com/research-data/?search=barknet">BarkNet</a>, une banque en donées ouvertes sous licence MIT de 23 000 photos d'écorces d'arbres en haute résolution prises avec des téléphones
intelligents par une équipe d'étudiants et de chercheurs du <a href="https://www.sbf.ulaval.ca/" target='_blank'>Département des sciences du bois et de la forêt de l'Université Laval</a> à Québec.</p>

## Licence

<hr style="line-height=2;"/>
Sous licence Apache License, Version 2.0 (la "Licence");

Vous ne pouvez pas utiliser ce fichier sauf en conformité avec la licence.

Vous pouvez obtenir une copie de la licence à: http://www.apache.org/licenses/LICENSE-2.0

Sauf si requis par la loi applicable ou convenu par écrit, le logiciel
distribué sous la Licence est distribué sur une BASE "TEL QUEL",
SANS GARANTIE OU CONDITION D'AUCUNE SORTE, expresse ou implicite.
Consultez la Licence pour connaître les autorisations et limitations
spécifiques à la langue dans le cadre de la Licence.
<hr style="line-height=2;"/>
Licensed under the Apache License, Version 2.0 (the "License");

You may not use this file except in compliance with the License.

You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
<hr style="line-height=2;"/>
