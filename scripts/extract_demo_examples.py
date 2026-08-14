"""Extrait 3 profils représentatifs du dataset d'entraînement pour peupler les
exemples préchargés de la démo Gradio (low risk / high risk / borderline).

Nécessite en local :
- data/app_train_enriched.parquet (référence drift du Projet 6)
- models/model.joblib

Écrit data/demo_examples.json, lu automatiquement par src/app.py au démarrage.

Usage :
    uv run python -m scripts.extract_demo_examples
"""
import json
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "BUREAU_BUREAU_DEBT_CREDIT_RATIO_MAX",
    "APP_PAYMENT_RATE",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE",
    "INSTAL_INSTAL_DAYS_LATE_MAX",
    "NAME_EDUCATION_TYPE_Higher education",
    "NAME_FAMILY_STATUS_Married",
    "PREV_PREV_APP_CREDIT_RATIO_MEAN",
    "POS_NB_ENTRIES",
    "DAYS_BIRTH",
    "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
    "INSTAL_INSTAL_PAYMENT_RATIO_MEAN",
    "PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN",
    "BUREAU_DAYS_ENDDATE_FACT_MAX",
    "INSTAL_NB_PAYMENTS",
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
TRAIN_PATH = DATA_DIR / "app_train_enriched.parquet"
OUTPUT_PATH = DATA_DIR / "demo_examples.json"

# Bornes de probabilité de défaut visées pour chaque profil
TARGETS = {
    "Low risk": (0.00, 0.20),
    "High risk": (0.75, 1.00),
    "Borderline, near threshold": (0.46, 0.50),
}


def to_native(v):
    """Convertit un scalaire numpy (float64, int64, bool_) en type Python natif.

    row.tolist() sur une Series à dtype mixte ne convertit pas les éléments,
    contrairement à un array numpy homogène : il faut le faire élément par élément.
    Les colonnes one-hot (bool) sont ramenées à 0/1 pour rester cohérentes avec
    la convention documentée (cf. Synthese_Projet6, README).
    """
    if pd.isna(v):
        return None
    if isinstance(v, (bool,)) or type(v).__name__ in ("bool", "bool_"):
        return int(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def pick_row(proba, low, high, used_indices):
    """Index de la ligne dont la proba est la plus proche du centre de la plage.

    proba est déjà restreint aux lignes complètes : la ligne retournée n'a donc
    aucune valeur manquante sur les 20 features, ce qui évite les champs vides
    dans les exemples préchargés de la démo.
    """
    mask = (proba >= low) & (proba <= high)
    candidates = proba.index[mask].difference(used_indices)
    if len(candidates) == 0:
        raise ValueError(
            f"Aucune ligne complète trouvée dans la plage [{low}, {high}]. "
            f"Élargir la plage correspondante dans TARGETS."
        )
    center = (low + high) / 2
    return (proba[candidates] - center).abs().idxmin()


def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(TRAIN_PATH, columns=FEATURES)

    # Seules les lignes intégralement renseignées sont éligibles : un exemple
    # préchargé avec un champ vide se lit comme un bug côté visiteur, même si
    # le pipeline sait imputer. Le modèle reste évalué sur les lignes retenues.
    complete = df.dropna(subset=FEATURES)
    print(f"{len(complete)} lignes complètes sur {len(df)} ({len(complete) / len(df):.1%})")
    if complete.empty:
        raise ValueError("Aucune ligne complète sur les 20 features dans le dataset.")

    proba = pd.Series(model.predict_proba(complete[FEATURES])[:, 1], index=complete.index)

    examples, labels, used = [], [], pd.Index([])
    for label, (low, high) in TARGETS.items():
        idx = pick_row(proba, low, high, used)
        used = used.append(pd.Index([idx]))
        row = complete.loc[idx, FEATURES]
        values = [to_native(v) for v in row]
        if any(v is None for v in values):
            raise ValueError(f"Valeur nulle inattendue dans l'exemple '{label}'")
        examples.append(values)
        labels.append(f"{label} (p={proba[idx]:.2f})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"examples": examples, "labels": labels}, f, indent=2)

    print(f"{len(examples)} exemples écrits dans {OUTPUT_PATH}")
    for label, ex in zip(labels, examples):
        print(f"  - {label}")


if __name__ == "__main__":
    main()