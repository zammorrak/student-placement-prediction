from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report


def _preparer_sortie(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def tracer_matrices_confusion(modeles, x_test, y_test, output_dir="outputs"):
    output_path = _preparer_sortie(output_dir)
    fig, axes = plt.subplots(1, len(modeles), figsize=(6 * len(modeles), 5))
    axes = np.atleast_1d(axes)

    for ax, (nom, modele) in zip(axes, modeles.items()):
        y_pred = modele.predict(x_test)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(nom)

    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tracer_courbes_roc(modeles, x_test, y_test, output_dir="outputs"):
    output_path = _preparer_sortie(output_dir)
    fig, ax = plt.subplots(figsize=(8, 6))

    for nom, modele in modeles.items():
        y_pred_proba = modele.predict_proba(x_test) if hasattr(modele, 'predict_proba') else modele.predict(x_test)
        if y_pred_proba.ndim == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
        try:
            RocCurveDisplay.from_predictions(y_test, y_pred_proba[:, 1], ax=ax, name=nom)
        except:
            pass

    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_title("Comparaison des courbes ROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tracer_importance_features(modeles, feature_names, output_dir="outputs"):
    output_path = _preparer_sortie(output_dir)

    modeles_supportes = {}
    for nom, modele in modeles.items():
        if hasattr(modele, "feature_importances_"):
            modeles_supportes[nom] = modele.feature_importances_
        elif hasattr(modele, "coef_"):
            modeles_supportes[nom] = np.abs(modele.coef_).ravel()

    if not modeles_supportes:
        return

    for nom, importances in modeles_supportes.items():
        idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=np.array(feature_names)[idx], y=np.array(importances)[idx], palette="viridis")
        plt.title(f"Importance des features - {nom}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        slug = re.sub(r"[^a-z0-9]+", "_", nom.lower()).strip("_")
        filename = f"feature_importance_{slug}.png"
        plt.savefig(output_path / filename, dpi=150, bbox_inches="tight")
        plt.close()


def sauvegarder_rapports_classification(modeles, x_test, y_test, output_dir="outputs"):
    output_path = _preparer_sortie(output_dir)
    rapport_path = output_path / "classification_reports.txt"

    with open(rapport_path, "w", encoding="utf-8") as file:
        for nom, modele in modeles.items():
            y_pred = modele.predict(x_test)
            report = classification_report(
                y_test,
                y_pred,
                target_names=["Non Placé", "Placé"],
                zero_division=0
            )
            file.write(f"{'=' * 20} {nom} {'=' * 20}\n")
            file.write(report)
            file.write("\n\n")
