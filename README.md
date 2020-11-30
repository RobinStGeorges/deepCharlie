#Projet DEEP LEARNING 5ALJ

* Robin SAINT GEORGES
* Allam HADJI
* Kevin TONDO

Ce projet a pour but de déterminer la position de Charlie dans une image de type "Où est Charlie?", en utilisant le deep learning et nottament le CNN.

Afin de faire  tourner ce programme, merci de suivre les étapes suivantes :

* >pip install -r requirements.txt
  
* ajouter le fichier "source.jpg" dans lequel vous voulez chercher charlie 
 
* Executer dataCreate.py
  * celà génère le fichier data.csv avec les correspondance nom de fichier, classe

* Executer CNN.py
  * Celà creer le fichier toto.dat, contenant les poids du cnn

* executer le fichier findCharlie.py 
  * La réponse sera crée dans le fichier "result.jpg"