import json

import gradio as gr
import joblib
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# Définition du seuil métier optimal déterminé au projet 6
SEUIL = 0.48

# Emplacement du fichier de logging
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Emplacement des exemples de démo (générés par scripts/extract_demo_examples.py)
DEMO_EXAMPLES_PATH = DATA_DIR / "demo_examples.json"

# Chargement du modèle
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
model_path = MODEL_DIR / "model.joblib"
model = joblib.load(model_path)

# Fonction de prédiction
def predict(
        ext_source_2: float,
        ext_source_3: float,
        ext_source_1: float,
        bureau_bureau_debt_credit_ratio_max: float,
        app_payment_rate: float,
        amt_annuity: float,
        days_employed: int,
        amt_goods_price: float,
        instal_instal_days_late_max: int,
        name_education_type_higher_education: int,
        name_family_status_married: int,
        prev_prev_app_credit_ratio_mean: float,
        pos_nb_entries: int,
        days_birth: int,
        prev_days_last_due_1st_version_max: int,
        pos_cnt_instalment_future_mean: float,
        instal_instal_payment_ratio_mean: float,
        prev_name_contract_status_refused_mean: float,
        bureau_days_enddate_fact_max: int,
        instal_nb_payments: int,
    ):

    # Création du DataFrame d'entrée
    df = pd.DataFrame(
        data=[[
            ext_source_2,
            ext_source_3,
            ext_source_1,
            bureau_bureau_debt_credit_ratio_max,
            app_payment_rate,
            amt_annuity,
            days_employed,
            amt_goods_price,
            instal_instal_days_late_max,
            name_education_type_higher_education,
            name_family_status_married,
            prev_prev_app_credit_ratio_mean,
            pos_nb_entries,
            days_birth,
            prev_days_last_due_1st_version_max,
            pos_cnt_instalment_future_mean,
            instal_instal_payment_ratio_mean,
            prev_name_contract_status_refused_mean,
            bureau_days_enddate_fact_max,
            instal_nb_payments,
        ]],
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
    # Démarrage timer (temps d'inférence)
    start_time = time.time()

    # Calcul de la prédiction de probabilité
    y_pred_proba = model.predict_proba(df)[0][1]

    # Calcul du temps d'inférence
    inference_time = time.time() - start_time

    # Décision en fonction du seuil
    message = "Crédit accordé" if y_pred_proba < SEUIL else "Crédit refusé"
    status = "OK"

    if df.isnull().sum().sum() > 10:
        message += " . Attention, un nombre faible de variable est entré. En ajoutant d'autres valeurs, le résultat sera plus précis"
        status = "INPUT_INCOMPLETE"
    
    # Remplissage du fichier de logging
    csv_path = DATA_DIR / "logging.csv"
    log_df = df.copy() # Copie du DataFrame d'entrée pour récupérer les 20 variables d'un coup
    log_df["Y_PRED_PROBA"] = y_pred_proba
    log_df["MESSAGE"] = message
    log_df["STATUS"] = status
    log_df["TIMESTAMP"] = datetime.now()
    log_df["INFERENCE_TIME"] = inference_time
    try:
        log_df.to_csv(path_or_buf=csv_path, index=False, mode="a", header=not(csv_path.exists()))
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier : {e}")

    return message, y_pred_proba


def load_demo_examples():
    """Charge les exemples précalculés par scripts/extract_demo_examples.py, s'ils existent."""
    if DEMO_EXAMPLES_PATH.exists():
        with open(DEMO_EXAMPLES_PATH) as f:
            payload = json.load(f)
        return payload.get("examples"), payload.get("labels")
    return None, None


demo_examples, demo_example_labels = load_demo_examples()

# Définition des variables en input
ext_source_2 = gr.Number(value=None, label="External normalised score 2", info="Between 0 and 1")
ext_source_3 = gr.Number(value=None, label="External normalised score 3", info="Between 0 and 1")
ext_source_1 = gr.Number(value=None, label="External normalised score 1", info="Between 0 and 1")
bureau_bureau_debt_credit_ratio_max = gr.Number(value=None, label="Max debt to credit ratio (credit bureau)")
app_payment_rate = gr.Number(value=None, label="Annual payment rate")
amt_annuity = gr.Number(value=None, label="Loan annuity amount", info="Currency units")
days_employed = gr.Number(value=None, label="Days in current job", info="Negative value")
amt_goods_price = gr.Number(value=None, label="Price of financed goods", info="Currency units")
instal_instal_days_late_max = gr.Number(value=None, label="Max late payment, in days")
name_education_type_higher_education = gr.Dropdown(choices=[("Yes", 1), ("No", 0)], label="Higher education?")
name_family_status_married = gr.Dropdown(choices=[("Yes", 1), ("No", 0)], label="Married?")
prev_prev_app_credit_ratio_mean = gr.Number(value=None, label="Mean approved / requested credit ratio (previous applications)")
pos_nb_entries = gr.Number(value=None, label="Number of late installments (POS)")
days_birth = gr.Number(value=None, label="Age in days", info="Negative value")
prev_days_last_due_1st_version_max = gr.Number(value=None, label="Max due date, first version (previous applications)")
pos_cnt_instalment_future_mean = gr.Number(value=None, label="Mean remaining installments (previous credit)")
instal_instal_payment_ratio_mean = gr.Number(value=None, label="Mean paid / due ratio (installments)")
prev_name_contract_status_refused_mean = gr.Number(value=None, label="Mean refusal rate (previous applications)")
bureau_days_enddate_fact_max = gr.Number(value=None, label="Days since last closed credit (bureau)")
instal_nb_payments = gr.Number(value=None, label="Number of past installment payments")

# Définition des entrées dans Gradio
demo = gr.Interface(
    fn=predict,
    inputs=[
        ext_source_2,
        ext_source_3,
        ext_source_1,
        bureau_bureau_debt_credit_ratio_max,
        app_payment_rate,
        amt_annuity,
        days_employed,
        amt_goods_price,
        instal_instal_days_late_max,
        name_education_type_higher_education,
        name_family_status_married,
        prev_prev_app_credit_ratio_mean,
        pos_nb_entries,
        days_birth,
        prev_days_last_due_1st_version_max,
        pos_cnt_instalment_future_mean,
        instal_instal_payment_ratio_mean,
        prev_name_contract_status_refused_mean,
        bureau_days_enddate_fact_max,
        instal_nb_payments,
    ],
    # Définition des sorties dans Gradio
    outputs=[gr.Textbox(label="Verdict"), gr.Number(label="Default probability")],
    examples=demo_examples,
    example_labels=demo_example_labels,
    title="Credit default scoring API",
    description=(
        "LightGBM pipeline predicting the probability of payment default from 20 features. "
        "Decision threshold set at 0.48, optimised on a business cost function weighting "
        "false negatives ten times more than false positives (FN x 10 + FP x 1). "
        "All fields are optional, missing values are median-imputed by the sklearn pipeline."
    ),
    article="Source: https://github.com/Jojo4911/credit-scoring-mlops",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")