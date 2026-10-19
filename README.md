# EndoPredict: AI-Powered Diabetes Risk Prediction

An end-to-end Machine Learning pipeline and web application designed to predict the probability of a patient having diabetes. The project is built using a Random Forest Classifier trained on the Kaggle Diabetes Prediction Dataset (100,000 patient records).

## Model Performance

The Random Forest model was trained using stratified splits and class-weight balancing to mitigate class imbalance (since only ~8.5% of the dataset is diabetic).

*   **Accuracy:** `91.82%`
*   **ROC-AUC Score:** `97.64%`
*   **Diabetic Class Recall:** `89%` (1511 out of 1700 true positive test instances)

### Feature Importance (Ranked)
1.  **HbA1c Level:** `42.07%`
2.  **Blood Glucose Level:** `28.07%`
3.  **Age:** `15.49%`
4.  **BMI:** `7.85%`
5.  **Hypertension / Heart Disease / Other Factors:** `~6.5%`

---

## File Structure

```text
Disease_Prediction/
├── diabetes_prediction_dataset.csv  # Raw Dataset
├── requirements.txt                 # Project dependencies
├── training.ipynb                   # Data preprocessing, training, & evaluation (Jupyter Notebook)
├── predict.py                       # CLI interactive prediction script
├── app.py                           # Flask backend web server
├── templates/
│   └── index.html                   # Beautiful frontend web layout
├── static/
│   ├── style.css                    # Modern CSS (Glassmorphism & animations)
│   └── script.js                    # Form validation & prediction gauge logic
├── diabetes_model_pipeline.joblib   # Saved preprocessor & model (Generated)
└── README.md                        # Documentation
```

---

## Setup and Running Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Run the following command in your terminal to install requirements:
```bash
pip install -r requirements.txt
```

### 2. Train the ML Model
Open and run all cells in the Jupyter Notebook:
`training.ipynb`

This notebook uses the `pd.csv` method (aliased to `pd.read_csv`) to load the dataset, preprocesses it, trains the Random Forest classifier, and saves the binary pipeline to `diabetes_model_pipeline.joblib`.

### 3. Predict via Command Line
You can test the model interactively using the CLI script. Run:
```bash
python predict.py
```
You will be prompted to enter patient features (Age, Gender, BMI, HbA1c, etc.) and it will display the risk percentage.

### 4. Run the Web Application
Launch the Flask development server:
```bash
python app.py
```
After starting, open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

You will see a gorgeous glassmorphism UI with animations, input validation, a visual speedometer risk-indicator gauge, and personalized health recommendations based on risk scores.
