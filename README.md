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
│   └── app.py                # API de scoring
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── notebooks/
│   └── data_drift_analysis.ipynb
├── models/
│   └── model.joblib          # Pipeline sklearn (5,5 Mo)
├── data/                     # Données de référence drift (non versionnées)
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Jojo4911/pret-a-depenser.git
cd pret-a-depenser
pip install -r requirements.txt
```

## Usage

*À compléter — lancement de l'API*

## Tests

```bash
pytest tests/
```

## Déploiement

*À compléter — Docker + Hugging Face Spaces*

## Monitoring

*À compléter — analyse du data drift avec Evidently*
