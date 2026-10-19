import os
import pandas as pd
import joblib

def get_input(prompt, type_func, validation_func=None, error_msg="Invalid input. Please try again."):
    while True:
        try:
            val = type_func(input(prompt).strip())
            if validation_func and not validation_func(val):
                print(error_msg)
                continue
            return val
        except ValueError:
            print(error_msg)

def main():
    model_path = "diabetes_model_pipeline.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Run 'train.py' first.")
        return

    print("Loading diabetes prediction model...")
    pipeline = joblib.load(model_path)
    print("Model loaded successfully.\n")
    print("--- Diabetes Risk Predictor CLI ---")
    print("Please enter patient details:")

    gender = get_input(
        "Gender (Male/Female/Other): ",
        str,
        lambda x: x.capitalize() in ["Male", "Female", "Other"],
        "Please enter 'Male', 'Female', or 'Other'."
    ).capitalize()

    age = get_input(
        "Age (0-120): ",
        float,
        lambda x: 0 <= x <= 120,
        "Please enter a valid age between 0 and 120."
    )

    hypertension = get_input(
        "Hypertension (1 = Yes, 0 = No): ",
        int,
        lambda x: x in [0, 1],
        "Please enter 0 or 1."
    )

    heart_disease = get_input(
        "Heart Disease (1 = Yes, 0 = No): ",
        int,
        lambda x: x in [0, 1],
        "Please enter 0 or 1."
    )

    smoking_options = ["never", "No Info", "current", "former", "ever", "not current"]
    print(f"Smoking options: {', '.join(smoking_options)}")
    smoking_history = get_input(
        "Smoking History: ",
        str,
        lambda x: x in smoking_options,
        f"Please choose from: {', '.join(smoking_options)}"
    )

    bmi = get_input(
        "BMI (5-100): ",
        float,
        lambda x: 5 <= x <= 100,
        "Please enter a valid BMI between 5 and 100."
    )

    hba1c = get_input(
        "HbA1c Level (3.0-15.0): ",
        float,
        lambda x: 3 <= x <= 15,
        "Please enter a valid HbA1c level between 3.0 and 15.0."
    )

    glucose = get_input(
        "Blood Glucose Level (30-500 mg/dL): ",
        int,
        lambda x: 30 <= x <= 500,
        "Please enter a valid blood glucose level between 30 and 500."
    )

    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose
    }])

    print("\nPredicting...")
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    print("\n--- Prediction Results ---")
    print(f"Diabetes Probability: {probability * 100:.2f}%")
    if prediction == 1:
        print("Result: DIABETIC DETECTED (High Risk)")
        print("Recommendation: Please consult a physician/endocrinologist for a diagnostic medical evaluation.")
    else:
        print("Result: NO DIABETES DETECTED (Low/Normal Risk)")
        print("Recommendation: Maintain a healthy diet, exercise, and regular checkups.")

if __name__ == "__main__":
    main()
