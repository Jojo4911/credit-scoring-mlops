"""
Génère des requêtes simulées vers predict() pour peupler logging.csv avec des données exploitables par l'analyse de drift.

Trois familles de requêtes :
- 80 requêtes normales : proches de la distribution d'entraînement.
- 50 requêtes driftées : décalage volontaire et ciblé sur AMT_ANNUITY et DAYS_EMPLOYED, les autres features restent dans leur plage normale.
- 20 requêtes avec valeurs manquantes : pour vérifier que le filtre STATUS == 'OK' du notebook de drift les écarte correctement.

Usage : depuis la racine du repo, uv run --group dev python scripts/simulate_production_data.py
"""
import random
from src.app import predict

random.seed(42)  # Reproductibilité


def sample_normal_row():
    """Une ligne dans des plages réalistes, cf. Synthese_Projet6_pour_Projet8.md section 5.4."""
    return dict(
        ext_source_2=random.gauss(0.5, 0.15),
        ext_source_3=random.gauss(0.5, 0.15),
        ext_source_1=random.gauss(0.5, 0.15),
        bureau_bureau_debt_credit_ratio_max=random.gauss(0.5, 0.2),
        app_payment_rate=random.gauss(0.08, 0.03),
        amt_annuity=random.gauss(27000, 8000),
        days_employed=int(random.gauss(-2000, 1500)),
        amt_goods_price=random.gauss(500000, 150000),
        instal_instal_days_late_max=int(random.gauss(5, 10)),
        name_education_type_higher_education=random.choice([0, 1]),
        name_family_status_married=random.choice([0, 1]),
        prev_prev_app_credit_ratio_mean=random.gauss(0.9, 0.2),
        pos_nb_entries=int(random.gauss(15, 8)),
        days_birth=int(random.gauss(-15000, 3500)),
        prev_days_last_due_1st_version_max=int(random.gauss(100, 60)),
        pos_cnt_instalment_future_mean=random.gauss(20, 10),
        instal_instal_payment_ratio_mean=random.gauss(1.0, 0.3),
        prev_name_contract_status_refused_mean=random.gauss(0.15, 0.1),
        bureau_days_enddate_fact_max=int(random.gauss(-100, 200)),
        instal_nb_payments=int(random.gauss(12, 6)),
    )


def sample_drifted_row():
    """Comme une ligne normale, mais AMT_ANNUITY et DAYS_EMPLOYED décalés volontairement."""
    row = sample_normal_row()
    row["amt_annuity"] = random.gauss(70000, 10000)        # Bien au-dessus de la moyenne train (~27k)
    row["days_employed"] = int(random.gauss(-200, 100))    # Emploi très récent, profil atypique
    return row


def sample_missing_row():
    """Une ligne normale avec environ la moitié des champs vidés (None)."""
    row = sample_normal_row()
    keys = list(row.keys())
    for key in random.sample(keys, k=11):  # > 10 NaN, déclenche STATUS = INPUT_INCOMPLETE
        row[key] = None
    return row


def run_batch(sampler, count, label):
    for i in range(count):
        row = sampler()
        predict(**row)
    print(f"{count} requêtes '{label}' envoyées.")


if __name__ == "__main__":
    run_batch(sample_normal_row, 80, "normales")
    run_batch(sample_drifted_row, 50, "driftées")
    run_batch(sample_missing_row, 20, "valeurs manquantes")
    print("Simulation terminée, voir data/logging.csv")