# Imports
from time import perf_counter
import pandas as pd
import joblib
from pathlib import Path

time_load_1 = perf_counter()
model_check = joblib.load(Path("models/model.joblib"))
time_load_2 = perf_counter()
print(f"Chargement du modèle : {(time_load_2 - time_load_1) * 1000:.1f} ms")

from src.app import predict, model

payload = [
    0.75,
    0.28,
    0.13,
    0.7,
    0.16,
    5000,
    2000,
    100000,
    27,
    1,
    1,
    0.43,
    23,
    -15000,
    150,
    50,
    0.33,
    7,
    -42,
    13
]

# Fonction predict()
time_predict_func_1 = perf_counter()
for i in range(10):
    message, y_pred_proba = predict(*payload)
time_predict_func_2 = perf_counter()
predict_funct_time = ((time_predict_func_2 - time_predict_func_1) / 10) * 1000

print(f"Temps d'appel de la fonction predict() : {predict_funct_time:.1f} ms")

# Calcul du temps d'inférence manuel
df = pd.DataFrame(
    data=[payload],
    columns=[
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
)
time_predict_proba_1 = perf_counter()
predict_proba = model.predict_proba(df)[0][1]
time_predict_proba_2 = perf_counter()

print(f"Temps de calcul de la probabilité de défaut : {(time_predict_proba_2 - time_predict_proba_1) * 1000:.1f} ms")