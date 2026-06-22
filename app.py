import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODELS
# -----------------------------


@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "Decision Tree": joblib.load("models/Decision_tree.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "SVM": joblib.load("models/support_vector_machine.pkl"),
        "XGBoost": joblib.load("models/XGboost.pkl")
    }

    preprocessor = joblib.load("models/preprocessor.pkl")
    return models, preprocessor


models, preprocessor = load_models()

st.set_page_config(
    layout="wide"
)
st.title('AI-Based Car Insurance Claim Prediction')
st.caption('smart system that will help insurance company to predict  the driver will file a claim or not')

# -----------------------------
# MODEL SELECTION (RADIO)
# -----------------------------
model_name = st.radio(  # user choose a model
    "Choose ML Model to use for predicting",
    list(models.keys())
)

model = models[model_name]  # get the model apk

st.subheader(f"Selected Model: {model_name}")

# -----------------------------
# INPUT FORM
# -----------------------------
with st.form("prediction_form"):  # to prevent the app from re-running on every input change, it collece the data onetime

    AGE = st.selectbox("Age", ["Young", "Adult", "Middle-aged", "Old"])
    GENDER = st.selectbox("Gender", ["male", "female"])
    RACE = st.selectbox("Race", ["majority", "minority"])
    MARRIED = st.selectbox("Married", ["0", "1"])
    CHILDREN = st.number_input("Children", 0, 10, 0)
    CREDIT_SCORE = st.number_input("Credit Score", 0.0, 1.0, 0.5)
    ANNUAL_MILEAGE = st.number_input("Annual Mileage", 0, 50000, 10000)
    DRIVING_EXPERIENCE = st.selectbox(
        "Driving Experience",
        ["Novice (0-9y)", "Intermediate (10-19y)", "Experienced (20-29y)", "Expert (30y+)"]
    )

    EDUCATION = st.selectbox("Education", ["none", "high school", "university"])
    INCOME = st.selectbox("Income", ["poverty", "working class", "middle class", "upper class"])

    VEHICLE_YEAR = st.selectbox("Vehicle Year", ["before 2015", "after 2015"])
    VEHICLE_TYPE = st.selectbox("Vehicle Type", ["sedan", "sports car"])

    VEHICLE_OWNERSHIP = st.selectbox("Vehicle Ownership", ["0", "1"])

    SPEEDING_VIOLATIONS = st.number_input("Speeding Violations", 0, 20, 0)
    DUIS = st.number_input("DUIs", 0, 10, 0)
    PAST_ACCIDENTS = st.number_input("Past Accidents", 0, 10, 0)
    POSTAL_CODE = st.selectbox(
        "Postal Code",
        [10238, 32765, 92101, 21217],
        help="Select from available postal codes"
    )

    submit = st.form_submit_button("Predict")

# -----------------------------
# PREDICTION
# -----------------------------
if submit:

    input_data = pd.DataFrame({
        "AGE": [AGE],
        "GENDER": [GENDER],
        "RACE": [RACE],
        "DRIVING_EXPERIENCE": [DRIVING_EXPERIENCE],
        "EDUCATION": [EDUCATION],
        "INCOME": [INCOME],
        "VEHICLE_YEAR": [VEHICLE_YEAR],
        "VEHICLE_TYPE": [VEHICLE_TYPE],
        "VEHICLE_OWNERSHIP": [VEHICLE_OWNERSHIP],
        "CREDIT_SCORE": [CREDIT_SCORE],
        "MARRIED": [MARRIED],
        "CHILDREN": [CHILDREN],
        "POSTAL_CODE": [POSTAL_CODE],
        "ANNUAL_MILEAGE": [ANNUAL_MILEAGE],
        "SPEEDING_VIOLATIONS": [SPEEDING_VIOLATIONS],
        "DUIS": [DUIS],
        "PAST_ACCIDENTS": [PAST_ACCIDENTS]
    })

    processed = preprocessor.transform(input_data)

    proba = model.predict_proba(processed)[0]  # initalize probability to add confidence level later
    prediction = model.predict(processed)[0]

    # 4. calculate percentage of confidence
    confidence = max(proba) * 100

    # 5. showing result
    st.write(f" Confidence: {confidence:.1f}%")
    if prediction == 1:
        st.error(" Will file a claim")
    else:
        st.success(" Will NOT file a claim")
