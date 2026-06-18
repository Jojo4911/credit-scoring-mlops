# --- Imports ---
import pytest
from src.app import predict
from unittest.mock import patch
import pandas as pd
from pathlib import Path

# --- Fixture pour le test valide ---
@pytest.fixture()
def valid_input():
    return (
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
    )


# --- Fixture de mock pour imposer un chemin temporaire au fichier CSV ---
# autouse=True : s'applique à tous les tests du fichier sans avoir besoin de le déclarer en paramètre, sauf si on veut accéder à tmp_path lui-même.
@pytest.fixture(autouse=True)
def path_csv(tmp_path):
    with patch("src.app.DATA_DIR", tmp_path):
        yield tmp_path


# Test de prédiction avec des valeurs valides
def test_prediction_valid_input(valid_input, path_csv):
    response_valid_input = predict(*valid_input)
    assert response_valid_input[0] == "Crédit refusé"
    assert response_valid_input[1] > 0.48  # Seuil métier optimisé
    assert response_valid_input[1] <= 1  # Respecte la borne haute
    created_file_path = Path(path_csv / "logging.csv")
    assert created_file_path.is_file() == True


# Test de prédiction avec quelques valeurs incorrectes
def test_prediction_incorrect_input(valid_input):
    incorrect_input = list(valid_input)
    incorrect_input[0] = "Hello"
    incorrect_input[4] = "world"
    with pytest.raises(ValueError):
        predict(*incorrect_input)


# Test de prédiction avec plus de 10 valeurs manquantes
def test_prediction_missing_input(valid_input):
    missing_input = list(valid_input)
    for i in range(11):
        missing_input[i] = None
    response_missing_input = predict(*missing_input)
    assert "Attention, un nombre faible de variable est entré." in response_missing_input[0]


# Test de prédiction avec peu de valeurs manquantes
def test_prediction_few_missing_input(valid_input):
    few_missing_input = list(valid_input)
    for i in range(5):
        few_missing_input[i] = None
    response_few_missing_input = predict(*few_missing_input)
    assert "Attention, un nombre faible de variable est entré." not in response_few_missing_input[0]


# Test de prédiction avec des valeurs aberrantes
def test_prediction_out_range_input(valid_input):
    out_range_input = list(valid_input)
    out_range_input[13] = 500
    out_range_input[18] = 1000
    response_out_range_input = predict(*out_range_input)
    assert response_out_range_input[0] == "Crédit refusé"


# --- Fixture de mock pour imposer des valeurs de probabilités basses ---
@pytest.fixture()
def low_proba():
    with patch("src.app.model.predict_proba") as mock_low_proba:
        mock_low_proba.return_value = [[0.53, 0.47]]  # Renvoie une probabilité de 0.47
        yield


# Test de crédit accordé avec une probabilité de 0.47
def test_thershold_approved_credit(valid_input, low_proba):
    response_threshold_approved = predict(*valid_input)
    assert response_threshold_approved[0] == "Crédit accordé"


# --- Fixture de mock pour imposer des valeurs de probabilités hautes ---
@pytest.fixture()
def high_proba():
    with patch("src.app.model.predict_proba") as mock_high_proba:
        mock_high_proba.return_value = [[0.51, 0.49]]  # Renvoie une probabilité de 0.49
        yield


# Test de crédit refusé avec une probabilité de 0.49
def test_thershold_refused_credit(valid_input, high_proba):
    response_threshold_refused = predict(*valid_input)
    assert response_threshold_refused[0] == "Crédit refusé"


# Test de format de sortie : la proba est une probabilité valide, la décision est l'une des deux chaînes attendues par l'interface.
def test_output_format(valid_input):
    response = predict(*valid_input)
    assert isinstance(response[1], float)
    assert 0 <= response[1] <= 1
    assert response[0] in ["Crédit accordé", "Crédit refusé"]