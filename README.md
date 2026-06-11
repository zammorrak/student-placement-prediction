# Prédiction de Placement Étudiant - Machine Learning

## Objectif
Construire un système complet de **classification binaire** pour prédire `placement_status` (Placed / Not Placed).

## Dataset
- **Fichier**: `student_dataset_10000_rows.csv`
- **Observations**: 10,000
- **Variables explicatives**: `study_hours`, `attendance`, `sleep_hours`, `internet_usage`, `assignments_completed`, `previous_score`, `exam_score`
- **Cible**: `placement_status`

## Structure du projet
```text
├── main.py
├── requirements.txt
├──modeles/
│   ├── LogisticRegressionCustom.py
│   ├── RandomForestCustom.py
│   └── SVM.py
├── notebooks/
│   └── student_placement_analysis.ipynb
├──outputs/
├── src/
│   ├── data_cleaning.py
│   ├── model_training.py
│   └── visualization.py
└── README.md
```

## Exigences couvertes
1. **Nettoyage des données**:
   - Traitement des valeurs manquantes (médiane/mode)
   - Détection et traitement des outliers (IQR + clipping)
   - Standardisation des features
   - Encodage de la cible (et one-hot si colonnes catégorielles)
2. **Modèles entraînés**:
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machine (SVM)
3. **Évaluation**:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Matrices de confusion
   - Classification report
4. **Visualisations**:
   - Matrices de confusion (tous les modèles)
   - Comparaison ROC
   - Importance des features (modèles supportés)
5. **Conclusion**:
   - Sélection automatique du meilleur modèle (F1)

## Installation
```bash
pip install -r requirements.txt
```

## Exécution
1. Ajouter le dataset dans `data/
2. Lancer:
```bash
python main.py
```
Les sorties (graphiques + rapports) sont enregistrées dans `outputs/`.
