# pret-a-depenser

Projet de scoring de prêt

## Arborescence du projet

projet-8-mlops/
├── src/
│   ├── __init__.py
│   └── app.py              # API de scoring
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── notebooks/
│   └── data_drift_analysis.ipynb
├── models/                  # modèle exporté (.joblib/.pkl)
├── data/                    # données de référence (si légères)
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── .gitignore
├── .env.example             # template sans secrets
├── requirements.txt
└── README.md