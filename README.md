# Système de Détection de Criminels

Ce projet est une application basée sur Flask qui utilise un réseau de neurones convolutionnel (CNN) pour la reconnaissance faciale et la détection de criminels. Le système peut identifier des individus figurant sur une liste noire (par exemple, des criminels). Le projet prend en charge les flux vidéo en direct, les fichiers vidéo téléchargés, et les images. Il envoie également une notification par e-mail lorsqu'un criminel est détecté et aussi Stocke les informations des individus détectés.

---

## Fonctionnalités

- **Détection en direct** : Détecte les visages en temps réel via la caméra du système.
- **Téléchargement de fichiers vidéo** : Permet aux utilisateurs de télécharger des vidéos pour la détection.
- **Reconnaissance via image** : Permet la reconnaissance faciale à partir d'images téléchargées.
- **Notifications par e-mail** : Envoie un e-mail aux administrateurs avec les détails de la détection.
- **Intégration MongoDB** : Stocke les informations des individus détectés (y compris leur image).
- **Seuil de confiance élevé** : Réduit les faux positifs grâce à un seuil de confiance configurable.

---

## Remarque sur les Données

Le projet fait face à une contrainte importante : **le manque de données visuelles disponibles sur les criminels sur le web**. Pour compenser ce problème, nous avons utilisé des techniques d'augmentation de données, telles que la **rotation**, le **zoom**, et d'autres transformations pour enrichir artificiellement notre dataset. Ces techniques permettent de mieux entraîner le modèle en diversifiant les angles et les conditions de prise de vue, augmentant ainsi sa robustesse.

---

## Prérequis

Avant de lancer l'application, assurez-vous d'avoir installé :

- Python 3.12.0
- Flask
- OpenCV
- TensorFlow/Keras
- MongoDB
- Une adresse e-mail pour les notifications (avec serveur SMTP configuré).


