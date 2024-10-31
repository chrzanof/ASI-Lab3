import pandas as pd
import numpy as np
import argparse
import os
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score


# Function to compute evaluation metrics
def evaluate_model(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mape, mae, mse, r2


# Function to visualize data distributions
def visualize_data(df):
    charts_dir = "charts"
    os.makedirs(charts_dir, exist_ok=True)

    for column in df.columns:
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.hist(df[column].dropna(), bins=30, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(charts_dir, f"{column}_histogram.png"))
        else:
            sns.countplot(y=df[column], palette='viridis')
            plt.title(f'Count Plot of {column}')
            plt.xlabel('Count')
            plt.ylabel(column)
            plt.savefig(os.path.join(charts_dir, f"{column}_countplot.png"))
        plt.close()


def main(data_path, save_path, n_folds, seed):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("model_training.log"), logging.StreamHandler()])
    logger = logging.getLogger()

    logger.info("Starting model training pipeline.")

    # Load the dataset
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    # Drop unmeaningful feature
    df = df.drop(columns=['rownames'], errors='ignore')

    # Visualize data distributions
    logger.info("Creating and saving histograms of dataset features.")
    visualize_data(df)

    # Separate target variable and features
    X = df.drop(columns=['score'])
    y = df['score']
    logger.info("Dataset loaded successfully. Splitting into features and target.")

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    models = {
        'GradientBoosting': GradientBoostingRegressor(random_state=seed),
        'XGBoost': XGBRegressor(random_state=seed),
        'RandomForest': RandomForestRegressor(random_state=seed),
        'MLPRegressor': MLPRegressor(random_state=seed, max_iter=1000)
    }

    param_grids = {
        'GradientBoosting': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.1, 0.05, 0.01],
            'regressor__max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.1, 0.05, 0.01],
            'regressor__max_depth': [3, 5, 7]
        },
        'RandomForest': {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 15, 20],
            'regressor__min_samples_split': [2, 5, 10]
        },
        'MLPRegressor': {
            'regressor__hidden_layer_sizes': [(50, 100, 50)],
            'regressor__activation': ['relu'],
            'regressor__alpha': [0.01],
            'regressor__learning_rate_init': [0.01, 0.1]
        }
    }

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Models will be saved to {save_path}.")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results_summary = []

    for model_name, model in models.items():
        logger.info(f"Starting training for {model_name} model.")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            cv=kfold,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        logger.info(f"Running grid search for {model_name}.")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

        y_train_cv_pred = cross_val_predict(best_model, X_train, y_train, cv=kfold, n_jobs=-1)
        mape_train, mae_train, mse_train, r2_train = evaluate_model(y_train, y_train_cv_pred)
        logger.info(
            f"{model_name} Cross-Validation on Train - MAPE: {mape_train:.4f}, MAE: {mae_train:.4f}, MSE: {mse_train:.4f}, R^2: {r2_train:.4f}")

        results_summary.append({
            'model': model_name,
            'dataset': 'Train (CV)',
            'mape': mape_train,
            'mae': mae_train,
            'mse': mse_train,
            'r2': r2_train
        })

        y_val_pred = best_model.predict(X_val)
        mape, mae, mse, r2 = evaluate_model(y_val, y_val_pred)
        logger.info(
            f"{model_name} Validation Results - MAPE: {mape:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")

        results_summary.append({
            'model': model_name,
            'dataset': 'Validation',
            'mape': mape,
            'mae': mae,
            'mse': mse,
            'r2': r2
        })

        model_path = os.path.join(save_path, f"{model_name}_model.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"{model_name} model saved to: {model_path}")

    logger.info("Starting evaluation on test set.")
    for model_name in models.keys():
        model_path = os.path.join(save_path, f"{model_name}_model.pkl")
        best_model = joblib.load(model_path)

        y_test_pred = best_model.predict(X_test)
        mape, mae, mse, r2 = evaluate_model(y_test, y_test_pred)
        logger.info(f"{model_name} Test Results - MAPE: {mape:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")

        results_summary.append({
            'model': model_name,
            'dataset': 'Test',
            'mape': mape,
            'mae': mae,
            'mse': mse,
            'r2': r2
        })

    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values(by=['model', 'dataset'])
    logger.info("\nSummary of Model Evaluation Results:")
    logger.info(results_df.reset_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models with grid search and cross-validation.")
    parser.add_argument("data_path", type=str, help="Path to the CSV file with the dataset.")
    parser.add_argument("save_path", type=str, help="Directory to save trained models.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args.data_path, args.save_path, args.n_folds, args.seed)
