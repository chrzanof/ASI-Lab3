# Lab3-Analizator_wynikow

## Opis projektu
Celem tego projektu jest opracowanie modeli regresyjnych, które przewidują wartość zmiennej `score` na podstawie wielu zmiennych, w tym demograficznych, ekonomicznych i edukacyjnych. Wykorzystano kilka modeli, takich jak Gradient Boosting, Random Forest, XGBoost i wielowarstwowy perceptron (MLP), które oceniono za pomocą grid search i kroswalidacji.


## Struktura plików
- `s24154.py`: Główny skrypt do uruchomienia procesu uczenia maszynowego i oceny modeli.
- `charts/`: Katalog zawierający wygenerowane histogramy i wykresy słupkowe.
- `model_training.log`: Plik z zapisanymi krokami i informacjami o procesie trenowania.
- `requirements.txt`: Plik wymagań do zainstalowania

## Przygotowanie i uruchomienie
### Wymagania wstępne
- Python >= 3.8
- pip 

### Instalacja
```bash
pip install -r requirements.txt
```

### Uruchomienie
```bash
python s24154.py <sciezka_do_danych.csv> <sciezka_do_zapisu_modeli> --n_folds <liczba_foli> --seed <ziarno>
```

## Analiza i inżynieria danych
### Analiza
![distance histogram](charts/distance_histogram.png)

### Inżynieria
- została usunięta kolumna `rownames`

## Wyniki i podsumowanie
Modele zostały ocenione przy użyciu kroswalidacji oraz podzielone na zbiory: treningowy, walidacyjny i testowy. Wyniki dla każdego modelu przedstawiają metryki, takie jak MAPE, MAE, MSE i R^2.

### Wyniki
Dla każdego modelu uzyskano poniższe wyniki:

| Model              | Zbiór         | MAE        | MSE        | R^2      |
|--------------------|---------------|------------|------------|----------|
| GradientBoosting    | Test          | 5.665896   | 49.336871  | 0.317022 |
| GradientBoosting    | Train (CV)    | 5.734156   | 49.964912  | 0.336743 |
| GradientBoosting    | Validation     | 6.045087   | 53.805321  | 0.331545 |
| MLPRegressor       | Test          | 6.027713   | 56.466550  | 0.218325 |
| MLPRegressor       | Train (CV)    | 6.012229   | 55.520003  | 0.263003 |
| MLPRegressor       | Validation     | 6.255663   | 58.880428  | 0.268494 |
| RandomForest       | Test          | 5.710428   | 51.248257  | 0.290563 |
| RandomForest       | Train (CV)    | 5.844986   | 52.089164  | 0.308545 |
| RandomForest       | Validation     | 6.011434   | 54.111517  | 0.327741 |
| XGBoost            | Test          | 5.657715   | 49.245949  | 0.318281 |
| XGBoost            | Train (CV)    | 5.724472   | 49.678166  | 0.340550 |
| XGBoost            | Validation     | 6.042898   | 53.889854  | 0.330495 |

### Wnioski
- **Najlepszy wynik osiągnął model Gradient Boosting**, jednak jego R^2 jest stosunkowo niski, co sugeruje, że model może wymagać dalszej optymalizacji.
- **Random Forest** i **XGBoost** również wykazały zbliżone wyniki, sugerując ich przydatność dla tego zbioru danych.
- **MLPRegressor** wypadł gorzej niż modele drzewiaste, co może wynikać z niewystarczającej ilości danych do nauki lub problemów z doborem hiperparametrów.



