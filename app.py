"""
Streamlit app: Cardiovascular Disease Risk Predictor

Run:
    streamlit run app.py

Expects model.pkl and feature_cols.pkl in the same directory
(produced by train_model.py).
"""

import joblib
import pandas as pd
import streamlit as st

from data_prep import add_engineered_features, clip_single_input

st.set_page_config(page_title="Cardio Risk Predictor", page_icon="🫀", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, feature_cols


model, feature_cols = load_model()

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.caption(
    "Trained on the cardio_train dataset with a tuned Random Forest. "
    "This is a demo/educational tool, not a medical device — see the note at the bottom."
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        ap_high = st.number_input("Systolic BP (AP_HIGH)", min_value=80, max_value=240, value=120)
        ap_low = st.number_input("Diastolic BP (AP_LOW)", min_value=40, max_value=190, value=80)

    with col2:
        cholesterol = st.selectbox(
            "Cholesterol", options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x],
        )
        glucose = st.selectbox(
            "Glucose", options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x],
        )
        smoke = st.selectbox("Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alcohol = st.selectbox("Alcohol intake", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        physical_activity = st.selectbox(
            "Physically active", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = {
        "AGE": age,
        "GENDER": gender,
        "HEIGHT": height,
        "WEIGHT": weight,
        "AP_HIGH": ap_high,
        "AP_LOW": ap_low,
        "CHOLESTEROL": cholesterol,
        "GLUCOSE": glucose,
        "SMOKE": smoke,
        "ALCOHOL": alcohol,
        "PHYSICAL_ACTIVITY": physical_activity,
    }
    raw = clip_single_input(raw)

    row = pd.DataFrame([raw])
    row = add_engineered_features(row)
    row = row[feature_cols]  # enforce exact training-time column order

    proba = model.predict_proba(row)[0, 1]
    pred = int(proba >= 0.5)

    st.divider()
    if pred == 1:
        st.error(f"**Elevated risk** — model estimates a {proba:.0%} probability of cardiovascular disease.")
    else:
        st.success(f"**Lower risk** — model estimates a {proba:.0%} probability of cardiovascular disease.")

    st.progress(min(max(proba, 0.0), 1.0))

    st.caption(
        "⚠️ Educational demo only. This model was trained on a public dataset for a "
        "class project and is not validated for clinical use. It should never inform "
        "an actual medical decision — see a doctor for that."
    )
