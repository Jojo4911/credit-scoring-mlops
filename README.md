# Prêt à Dépenser — API de Scoring Crédit

Déploiement en production d'un modèle de scoring crédit pour l'entreprise "Prêt à Dépenser". Le modèle prédit la probabilité de défaut de paiement d'un client à partir de 20 features.

## Modèle

- **Algorithme** : LightGBM (pipeline sklearn : SimpleImputer → MinMaxScaler → LGBMClassifier)
- **Features** : 20 variables numériques (scores externes, historique crédit, données client)
- **Seuil de décision** : 0.48 (optimisé via fonction de coût métier : FN×10 + FP×1)
- **Origine** : Projet 6 — MLflow `Credit_Scoring_Model@champion` (version 60)

## Arborescence

```
├── src/
│   ├── __init__.py
│   └── app.py                # API Gradio de scoring
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── notebooks/
│   └── data_drift_analysis.ipynb
├── models/
│   └── model.joblib          # Pipeline sklearn (5,5 Mo)
├── data/                     # Données de production et référence drift (non versionnées)
│   └── logging.csv           # Logs des requêtes API (généré automatiquement)
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── .gitignore
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

```bash
git clone https://github.com/Jojo4911/pret-a-depenser.git
cd pret-a-depenser
uv sync
```

## Usage

Lancer l'API en local :

```bash
python src/app.py
```

L'interface Gradio s'ouvre dans le navigateur. L'API accepte les 20 features suivantes (toutes optionnelles — les valeurs manquantes sont imputées par la médiane via le pipeline sklearn) :

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `EXT_SOURCE_2` | float | Score externe normalisé n°2 |
| 2 | `EXT_SOURCE_3` | float | Score externe normalisé n°3 |
| 3 | `EXT_SOURCE_1` | float | Score externe normalisé n°1 |
| 4 | `BUREAU_BUREAU_DEBT_CREDIT_RATIO_MAX` | float | Ratio dette/crédit max (bureau de crédit) |
| 5 | `APP_PAYMENT_RATE` | float | Taux de remboursement annuel |
| 6 | `AMT_ANNUITY` | float | Montant de l'annuité |
| 7 | `DAYS_EMPLOYED` | int | Nombre de jours de l'emploi actuel |
| 8 | `AMT_GOODS_PRICE` | float | Prix du bien financé |
| 9 | `INSTAL_INSTAL_DAYS_LATE_MAX` | int | Retard maximum en jours (versements) |
| 10 | `NAME_EDUCATION_TYPE_Higher education` | binaire (0/1) | Études supérieures |
| 11 | `NAME_FAMILY_STATUS_Married` | binaire (0/1) | Marié(e) |
| 12 | `PREV_PREV_APP_CREDIT_RATIO_MEAN` | float | Ratio moyen montant accordé/demandé |
| 13 | `POS_NB_ENTRIES` | int | Nombre d'entrées POS |
| 14 | `DAYS_BIRTH` | int | Âge en jours (valeur négative) |
| 15 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | int | Échéance max première version (demandes précédentes) |
| 16 | `POS_CNT_INSTALMENT_FUTURE_MEAN` | float | Moyenne versements restants (crédit précédent) |
| 17 | `INSTAL_INSTAL_PAYMENT_RATIO_MEAN` | float | Ratio moyen payé/dû |
| 18 | `PREV_NAME_CONTRACT_STATUS_REFUSED_MEAN` | float | Proportion moyenne de demandes refusées |
| 19 | `BUREAU_DAYS_ENDDATE_FACT_MAX` | int | Jours depuis clôture du dernier crédit (bureau) |
| 20 | `INSTAL_NB_PAYMENTS` | int | Nombre de paiements passés (crédits précédents) |

**Sorties** :
- **Verdict** : "Crédit accordé" (probabilité < 0.48) ou "Crédit refusé" (probabilité ≥ 0.48)
- **Probabilité de défaut de paiement** : valeur entre 0 et 1

Un avertissement s'affiche si plus de 10 features sont manquantes.

## Logging

Chaque requête est enregistrée automatiquement dans `data/logging.csv` avec :
- Les 20 features d'entrée
- La probabilité prédite
- La décision (accordé/refusé)
- Le timestamp
- Le temps d'inférence du modèle (en secondes)

Ces logs servent de base pour l'analyse du data drift et le monitoring des performances.

## Tests

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Couverture actuelle : 94% (lignes non couvertes : gestion d'erreur d'écriture CSV, point d'entrée `__main__`).

## Déploiement

*À compléter — Docker + Hugging Face Spaces*

## Monitoring

*À compléter — analyse du data drift avec Evidently*
