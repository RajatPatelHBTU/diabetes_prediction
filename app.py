import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "diabetes_model_pipeline.joblib"
pipeline = None

if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model on startup: {e}")

@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "font-src 'self' https://fonts.gstatic.com; "
        "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self';"
    )
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            try:
                pipeline = joblib.load(MODEL_PATH)
            except Exception as e:
                return jsonify({'error': f"Failed to load model: {str(e)}"}), 500
        else:
            return jsonify({'error': "Model not found. Please run train.py first."}), 500

    try:
        data = request.get_json(force=True)
        gender = data.get('gender')
        age = float(data.get('age', 0))
        hypertension = int(data.get('hypertension', 0))
        heart_disease = int(data.get('heart_disease', 0))
        smoking_history = data.get('smoking_history')
        bmi = float(data.get('bmi', 0))
        hba1c = float(data.get('HbA1c_level', 0))
        glucose = float(data.get('blood_glucose_level', 0))

        if not gender or gender not in ['Female', 'Male', 'Other']:
            return jsonify({'error': 'Invalid gender. Must be Female, Male, or Other.'}), 400
        if age < 0 or age > 120:
            return jsonify({'error': 'Invalid age. Must be between 0 and 120.'}), 400
        if hypertension not in [0, 1]:
            return jsonify({'error': 'Invalid hypertension.'}), 400
        if heart_disease not in [0, 1]:
            return jsonify({'error': 'Invalid heart disease.'}), 400
        
        smoking_options = ["never", "No Info", "current", "former", "ever", "not current"]
        if not smoking_history or smoking_history not in smoking_options:
            return jsonify({'error': 'Invalid smoking history.'}), 400
        
        if bmi <= 0 or bmi > 100:
            return jsonify({'error': 'Invalid BMI.'}), 400
        if hba1c <= 0 or hba1c > 20:
            return jsonify({'error': 'Invalid HbA1c level.'}), 400
        if glucose <= 0 or glucose > 1000:
            return jsonify({'error': 'Invalid blood glucose level.'}), 400

        input_df = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose
        }])

        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])

        recommendations = []
        risk_level = "Low"
        if probability < 0.2:
            risk_level = "Low"
            recommendations.extend([
                "Continue maintaining a balanced diet and regular physical activity.",
                "Get routine health check-ups annually."
            ])
        elif probability < 0.5:
            risk_level = "Moderate"
            recommendations.extend([
                "Monitor blood sugar levels occasionally, especially if you have a family history.",
                "Consider consulting a doctor to optimize your BMI and dietary habits.",
                "Engage in regular moderate exercise (150 mins per week)."
            ])
        else:
            risk_level = "High"
            recommendations.extend([
                "It is highly recommended to consult a primary care physician or endocrinologist.",
                "Perform a diagnostic oral glucose tolerance test or fasting plasma glucose test.",
                "Work on lifestyle modifications, including a low-glycemic diet and weight management."
            ])

        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
