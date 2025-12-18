# 🫀 Prédiction des Maladies Cardiaques par Machine Learning  

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Description

Ce projet utilise des algorithmes de **Machine Learning** pour prédire la présence de maladies cardiaques chez des patients à partir de données cliniques et démographiques standardisées.  
L’objectif est de fournir un outil fiable, interprétable et interactif pouvant assister les professionnels de santé dans la prise de décision.

---

## 🎯 Objectifs

- Atteindre une précision de prédiction supérieure à **85 %**
- Minimiser les **faux négatifs**, représentant un risque clinique majeur
- Fournir un modèle **interprétable** pour les professionnels de santé
- Développer une interface web simple et interactive

---

## 📊 Dataset

- **Source** : Kaggle – *Heart Disease Dataset*
- **Nombre d’observations** : 1 026 patients
- **Caractéristiques** : 13 variables cliniques et démographiques
- **Variable cible** :
  - `0` : Patient sain
  - `1` : Patient atteint d’une maladie cardiaque

---

## 🚀 Fonctionnalités

- ✅ Nettoyage et préparation des données  
- ✅ Analyse exploratoire des données (EDA)  
- ✅ Entraînement de plusieurs modèles de Machine Learning  
- ✅ Évaluation des performances des modèles  
- ✅ Interface web interactive avec **Streamlit**  
- ✅ Visualisations (matrice de confusion, courbe ROC)  
- ✅ Sauvegarde et chargement des modèles entraînés  

---

## 📈 Performances du Modèle

| Modèle            | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------------|----------|-----------|--------|----------|---------|
| SVM Polynomial   | **92.68 %** | **97.00 %** | **97.00 %** | **93.00 %** | **0.98** |

---

## 🛠️ Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Installation

```bash
git clone https://github.com/votre-username/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```
---

## ▶️ Utilisation
### Lancer l’interface Web (Streamlit)
```bash 
streamlit run src/app.py
```

### Entraîner le modèle
```bash
python src/model.py
```

### Notebook Jupyter

- notebook/model.ipynb

## 📁 Structure du Projet
``` 
Heart-Disease-Prediction/
│
├── dataset/          # Données brutes
├── src/              # Code source
├── model/            # Modèles sauvegardés
├── notebooks/        # Notebooks d'analyse
├── images/           # Visualisations
├── app.py            # Application Streamlit
├── requirements.txt  # Entraînement du modèle
└── README.md         # README file
```
---

## 🖥️ Interface Streamlit

### L’application permet :

- Saisie manuelle des caractéristiques du patient
- Prédiction en temps réel
- Visualisation des facteurs influents
- Téléchargement des résultats

---

## 📊 Résultats
### Matrice de Confusion
- images/confusion_matrix_svm_poly.png

### Courbe ROC
- images/roc_curve_svm_poly.png

## 🧪 Technologies Utilisées

- Python

- Scikit-learn

- Streamlit

- Pandas / NumPy

- Matplotlib / Seaborn

- Joblib

---

## 📚 Références

Kaggle – Heart Disease Dataset

Documentation officielle Scikit-learn

---

## 👥 Contributeurs

- Basma El kadri
- Imane Baychou
- Ghita Benlachen

---

## 📄 Licence

Ce projet est sous licence MIT.
Voir le fichier LICENSE pour plus de détails.