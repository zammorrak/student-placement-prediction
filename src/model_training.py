import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from Modeles.LogisticRegressionCustom import LogisticRegressionCustom
from Modeles.RandomForestcustom import RandomForestcustom
from Modeles.SVM import LinearSVMCustom
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings
warnings.filterwarnings('ignore')

def entrainer_modeles(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n Données divisées:")
    print(f"  - Train set: {X_train.shape[0]} échantillons")
    print(f"  - Test set: {X_test.shape[0]} échantillons")
    
    # Modèles à entraîner
    modeles = {
        'Logistic Regression (Custom)': LogisticRegressionCustom(lr=0.1, n_iters=1500),
        'Random Forest (Custom)': RandomForestcustom(n_estimators=5, max_depth=5),
        'Support Vector Machine (Custom)': LinearSVMCustom(lr=0.001, n_iters=1000)
    }
    
    modeles_entraines = {}
    
    print(f"\n Entraînement de {len(modeles)} modèles...\n")
    
    for nom, modele in modeles.items():
        print(f"{nom}:")
        time_start = datetime.now()
        modele.fit(X_train, y_train)
        modeles_entraines[nom] = modele
        time_end = datetime.now()
        y_pred = modele.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}\n")
        print(f"  Temps d'entraînement: {(time_end - time_start).total_seconds():.4f} secondes\n")
    
    return modeles_entraines, X_train, X_test, y_train, y_test

def valider_modeles(modeles, X, y, cv=5):
    resultats = {}
    
    for nom, modele in modeles.items():
        print(f"➜ {nom}:")
        
        # Cross-validation
        scores = cross_val_score(modele, X, y, cv=cv, scoring='f1')
        
        print(f"  Scores par fold: {scores.round(4)}")
        print(f"  Moyenne: {scores.mean():.4f} (+/- {scores.std():.4f})\n")
        
        resultats[nom] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return resultats

def selectionner_meilleur_modele(modeles, X_test, y_test):
    meilleures_scores = {}
    
    for nom, modele in modeles.items():
        y_pred = modele.predict(X_test)
        score = f1_score(y_test, y_pred, zero_division=0)
        meilleures_scores[nom] = score
    
    meilleur_nom = max(meilleures_scores, key=meilleures_scores.get)
    meilleur_modele = modeles[meilleur_nom]
    meilleur_score = meilleures_scores[meilleur_nom]
    
    print(f"Meilleur modèle: {meilleur_nom}")
    print(f"F1-Score: {meilleur_score:.4f}\n")

    print("Classement:")
    for i, (nom, score) in enumerate(sorted(meilleures_scores.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {nom}: {score:.4f}")

    return meilleur_nom, meilleur_modele, meilleur_score
