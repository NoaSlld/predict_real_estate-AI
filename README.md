# Projet de Modélisation Mathématique - Prédiction des Prix de l'Immobilier

## Description
Ce projet fait seul en 7 jours dans le cadre de ma 3ème année de BUT vise à prédire les prix de l'immobilier en France en utilisant plusieurs modèles de machine learning.

L'application permet d'entraîner ces modèles sur l'historiques des transactions immobilière en France et d'estimer la valeur foncière d'un bien en fonction de sa localisation et de sa surface.

Ce dépot est fonctionelle mais non fini car encore imprécis.

!! Attention !! Les fichiers de données sont presque vide pour limiter la taille du dépot. Veuillez remplacer les fichier  `data/full_20XX.csv` par ceux de ce dataset Kaggle: https://www.kaggle.com/datasets/nechbamohammed/real-estate-dataset/data 

## Installation
### Prérequis
Avant de commencer, assurez-vous d'avoir installé les dépendances requises :

- matplotlib
- seaborn
- pandas
- scikit-learn
- joblib
- lightgbm
- numpy
- os (module standard Python, pas besoin d'installation)
- datetime (module standard Python, pas besoin d'installation)

```bash
pip install matplotlib seaborn pandas scikit-learn joblib lightgbm numpy
```
La Version de python utilisée lors du développement est Python 3.11.2

### Structure du projet
```
Projet_Modelisation_Mathematiques/
│-- data/                  # Dossier contenant les datasets et modèles sauvegardés
│-- src/                   
│   │-- analyse.py         # Visualisation et analyse des données
│   │-- evaluation.py      # Fonction d'évaluation des modèles
│   │-- input.py           # Gestion des entrées utilisateur
│   │-- loads.py           # Chargement et prétraitement des données
│   │-- models.py          # Implémentation des modèles
│   │-- nettoyage.py       # Script de nettoyage des données
│   │-- prediction.py      # Fonction de prédiction
│-- main.py                # Script principal
│-- README.md              
```

## Utilisation
### Nettoyage des Données
Avant d'entraîner les modèles, assurez-vous que les données sont nettoyées :
```bash
python3 src/nettoyage.py
```
Les données nettoyées seront stockées dans `data/full_cleaned.csv`  

### Exécution du Programme
Lancer le script principal depuis la racine du projet pour entraîner les modèles et faire des prédictions :
```bash
python3 main.py
```

Le programme :
1. Charge et prépare les données.
2. Entraîne les modèles de régression.
3. Demande à l'utilisateur des informations sur le bien immobilier.
4. Prédit le prix du bien à l'aide des modèles entraînés.

## Modèles Implémentés
- **Régression Linéaire** 
- **Random Forest** 
- **LightGBM** 

## Évaluation des Modèles
Les modèles sont évalués avec les métriques suivantes :
- **R²** (Coefficient de détermination)
- **MAE** (Erreur Absolue Moyenne)
- **MSE** (Erreur Quadratique Moyenne)
- **RMSE** (Racine de l'Erreur Quadratique Moyenne)

## Visualisation des Données
Des graphiques sont disponibles pour analyser les tendances des prix immobiliers :
- **Répartition du montant des transactions**
- **Évolution des prix moyens et médians par année**
- **Communes avec les prix au m² les plus élevés**
- **Top 10 des plus fortes augmentations de prix**
Pour les générer, il suffit de décommenter l'appel à la fonction displayGraphs() dans ```main.py```.

## Variables du code

- Les variables `X_train / X_test / X_val` représente les données d'entrée, soit les variables explicatives utilisées par le modèle pour apprendre et prédire.
- Les variables `y_train / Y_test / Y_val` représente les cibles, soit les valeurs que nous essayons de prédire.