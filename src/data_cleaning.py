import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def charger_donnees(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def analyser_donnees_manquantes(df):
    donnees_manquantes = df.isna().sum()
    if donnees_manquantes.sum() > 0:
        print("\nValeurs manquantes détectées:")
        print(donnees_manquantes[donnees_manquantes > 0])
    else:
        print("\nAucune valeur manquante")
    return donnees_manquantes

def traiter_valeurs_manquantes(df):
    df_clean = df.copy()
    colonnes_numeriques = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    colonnes_categorielles = [
        col for col in df_clean.columns if col not in colonnes_numeriques
    ]

    for col in colonnes_numeriques:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in colonnes_categorielles:
        if df_clean[col].isna().any():
            mode = df_clean[col].mode()
            valeur = mode.iloc[0] if not mode.empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(valeur)

    print("Valeurs manquantes traitées (médiane/mode)")
    return df_clean

def detecter_outliers(df, colonnes_numeriques):
    print("\n Détection des outliers (méthode IQR):")
    outliers_par_colonne = {}

    for col in colonnes_numeriques:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outliers_par_colonne[col] = n_outliers
        print(f"  - {col}: {n_outliers} outliers détectés")

    print(f"Total: {sum(outliers_par_colonne.values())} outliers détectés")
    return outliers_par_colonne

def traiter_outliers(df, colonnes_numeriques):
    df_clean = df.copy()
    for col in colonnes_numeriques:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    print("Outliers traités par clipping IQR")
    return df_clean

def standardiser_donnees(x):
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(
        scaler.fit_transform(x), columns=x.columns, index=x.index
    )
    print(" Données standardisées (moyenne=0, std=1)")
    return x_scaled, scaler

def encoder_cible(df, colonne_cible):
    le = LabelEncoder()
    y = le.fit_transform(df[colonne_cible])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Variable cible encodée: {mapping}")
    return pd.Series(y, name=colonne_cible, index=df.index), le