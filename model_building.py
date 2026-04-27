import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def build_model(data_path):
    # 1. Load Data
    df = pd.read_csv(data_path)

    # 2. Data Cleaning
    df = df.drop_duplicates()

    # 3. Label Encoding
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include='object').columns

    # In a production script, we handle categorical mapping carefully
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col])
    
    # Save encoder
    joblib.dump(le, 'label_encoder.pkl')

    # 4. Final Cleanup
    df_cleaned = df.dropna()
    X = df_cleaned.drop('Salary', axis=1)
    y = df_cleaned['Salary']

    # 5. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 6. Save Model
    joblib.dump(model, 'random_forest_regressor.pkl')
    print("Model and Encoder saved successfully.")

if __name__ == '__main__':
    build_model('/content/Salary_Data.csv')
