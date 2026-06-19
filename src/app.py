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

    # Création du DataFrame d’entrée
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
    # Démarrage timer (temps d’inférence)
    start_time = time.time()

    # Calcul de la prédiction de probabilité
    y_pred_proba = model.predict_proba(df)[0][1]

    # Calcul du temps d’inférence
    inference_time = time.time() - start_time

    # Décision en fonction du seuil
    message = "Crédit accordé" if y_pred_proba < SEUIL else "Crédit refusé"
    status = "OK"

    if df.isnull().sum().sum() > 10:
        message += " . Attention, un nombre faible de variable est entré. En ajoutant d'autres valeurs, le résultat sera plus précis"
        status = "INPUT_INCOMPLETE"
    
    # Remplissage du fichier de logging
    csv_path = DATA_DIR / "logging.csv"
    log_df = df.copy() # Copie du DataFrame d’entrée pour récupérer les 20 variables d’un coup
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

# Définition des variables en input
ext_source_2 = gr.Number(value=None, label="Veuillez entrer le score externe normalisé numéro 2")
ext_source_3 = gr.Number(value=None, label="Veuillez entrer le score externe normalisé numéro 3")
ext_source_1 = gr.Number(value=None, label="Veuillez entrer le score externe normalisé numéro 1")
bureau_bureau_debt_credit_ratio_max = gr.Number(value=None, label="Veuillez entrer le ratio entre la dette et le montant du crédit (Bureau du Crédit)")
app_payment_rate = gr.Number(value=None, label="Veuillez entrer le taux de remboursement annuel")
amt_annuity = gr.Number(value=None, label="Veuillez entrer l'annuité du Bureau de crédit")
days_employed = gr.Number(value=None, label="Veuillez entrer le nombre de jours de l'emploi actuel")
amt_goods_price = gr.Number(value=None, label="Veuillez entrer le prix des biens pour lesquels le crédit est accordé")
instal_instal_days_late_max = gr.Number(value=None, label="Veuillez entrer le retard maximum en jours")
name_education_type_higher_education = gr.Dropdown(choices=[("Oui", 1), ("Non", 0)], label="Avez-vous fait des études supérieures ?")
name_family_status_married = gr.Dropdown(choices=[("Oui", 1), ("Non", 0)], label="Êtes-vous marié ?")
prev_prev_app_credit_ratio_mean = gr.Number(value=None, label="Veuillez entrer le ratio de montant accordé / montant demandé")
pos_nb_entries = gr.Number(value=None, label="Veuillez entrer le nombre de mensualités en retard")
days_birth = gr.Number(value=None, label="Veuillez entrer votre âge en nombre de jours")
prev_days_last_due_1st_version_max = gr.Number(value=None, label="Veuillez entrer quand la première échéance de la demande précédente a-t-elle eu lieu ?")
pos_cnt_instalment_future_mean = gr.Number(value=None, label="Veuillez entrer la moyenne des versements restants à effectuer sur le crédit précédent")
instal_instal_payment_ratio_mean = gr.Number(value=None, label="Veuillez entrer la moyenne du ratio payé / dû")
prev_name_contract_status_refused_mean = gr.Number(value=None, label="Veuillez entrer la moyenne d'offres refusées au cours du mois")
bureau_days_enddate_fact_max = gr.Number(value=None, label="Veuillez entrer le nombre de jours écoulés depuis la clôture du crédit CB au moment de la demande auprès de Home Credit (uniquement pour les crédits clôturés)")
instal_nb_payments = gr.Number(value=None, label="Veuillez entrer le nombre de paiement passés pour des crédits précédents")

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
    outputs=[gr.Textbox(label="Verdict :"), gr.Number(label="Probabilité de défaut de paiement :")]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")