import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def main():
    dataset_path = "diabetes_prediction_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset '{dataset_path}' not found.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    cat_cols = ['gender', 'smoking_history']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=42
        ))
    ])

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"Training completed. Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    model_filename = "diabetes_model_pipeline.joblib"
    print(f"Saving model pipeline to {model_filename}...")
    joblib.dump(pipeline, model_filename)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
