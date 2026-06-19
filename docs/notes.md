## Warning sklearn : feature names

`predict_proba()` génère un UserWarning ("X does not have valid feature names") à chaque appel.

Confirmé présent dès le chargement MLflow au Projet 6 (cf. export_champion.ipynb), donc hérité du pipeline entraîné, pas introduit par l'API du Projet 8. Le MinMaxScaler du pipeline retourne du numpy en interne, perdant les noms de colonnes avant le LGBMClassifier.

Sans impact : probabilité identique et stable entre la version MLflow et la version .joblib exportée. Correction possible uniquement en ré-entraînant le pipeline avec `set_output(transform="pandas")`, hors périmètre du Projet 8.

## Déploiement Hugging Face : LFS classique rejeté, bascule vers huggingface_hub

Le déploiement initial utilisait `git push` direct vers le repo Git du Space HF, avec Git LFS pour `model.joblib` (5,5 Mo). Ce push échouait avec une erreur serveur HF explicite : "Your push was rejected because it contains binary
files. Please use Xet to store binary files."

Hypothèse confirmée : HF impose désormais Xet (leur évolution de LFS) au delà d'un certain seuil de taille de fichier binaire, plutôt que LFS classique. Un projet précédent (Projet 5) avec un modèle de 242 Ko n'avait jamais rencontré ce problème, restant sous ce seuil.

Solution retenue : remplacer le `git push hf main` par un upload via le client officiel `huggingface_hub` (`HfApi.upload_folder()`), qui gère Xet nativement et ne dépend plus du protocole Git pour le transfert du modèle. Le pipeline
CI/CD installe `huggingface_hub` et appelle l'API directement, sans manipuler de remote Git vers HF.

Effet secondaire découvert en cours de route : `actions/checkout@v4` avec `lfs: true` seul ne garantit pas le téléchargement réel du contenu LFS sur certains runners (smudge silencieusement incomplet, fichier récupéré comme
pointeur texte de 132 octets au lieu du binaire). Un `git lfs pull` explicite après le checkout est nécessaire pour forcer le téléchargement réel.

## Stratégie de collecte des données de prod

Chaque requête à `predict()` qui aboutit à un calcul de probabilité est journalisée dans `data/logging.csv` : 20 features d'entrée, probabilité prédite, décision, statut (OK / INPUT_INCOMPLETE), timestamp, temps d'inférence.

Limite connue : une requête dont le typage des entrées est invalide lève une ValueError avant l'écriture du log. Ces erreurs ne sont donc pas visibles dans logging.csv. Le taux d'erreur mesurable depuis ce fichier est donc un taux d'erreur applicatif (statut INPUT_INCOMPLETE), pas un taux d'erreur de saisie. Hors périmètre de correction pour ce projet.