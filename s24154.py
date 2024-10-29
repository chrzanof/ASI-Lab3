import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

file_path = 'CollegeDistance.csv'
data = load_dataset(file_path)

if data is not None:
    df = pd.DataFrame(data)
    df_cleaned = df.dropna()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_cleaned, columns=['gender', 'ethnicity', 'fcollege', 'mcollege', 
                                                      'home', 'urban', 'education', 'income', 'region'], drop_first=True)

    # Normalize the numerical features
    scaler = MinMaxScaler()
    numeric_features = ['wage', 'distance', 'tuition']
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

    summary = df_encoded.describe(include='all')
    missing_values = df.isnull().sum().describe()

    print("Summary of normalized data:")
    print(summary)

    print("\nMissing values in original data:")
    print(missing_values)

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.pairplot(df_encoded, diag_kind='kde')
    plt.suptitle('Pair Plot of Normalized Data', y=1.02)
    plt.show()

    df_encoded.hist(bins=20, figsize=(15, 10))
    plt.suptitle('Histograms of Normalized Features')
    plt.show()

    # Split the data into training and testing sets
    X = df_encoded.drop('score', axis=1)  # Target variable is 'score'
    y = df_encoded['score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

else:
    print("No data loaded.")
