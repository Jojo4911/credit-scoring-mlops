## Warning sklearn : feature names

`predict_proba()` génère un UserWarning ("X does not have valid feature names") à chaque appel.

Confirmé présent dès le chargement MLflow au Projet 6 (cf. export_champion.ipynb), donc hérité du pipeline entraîné, pas introduit par l'API du Projet 8. Le MinMaxScaler du pipeline retourne du numpy en interne, perdant les noms de colonnes avant le LGBMClassifier.

Sans impact : probabilité identique et stable entre la version MLflow et la version .joblib exportée. Correction possible uniquement en ré-entraînant le pipeline avec `set_output(transform="pandas")`, hors périmètre du Projet 8.