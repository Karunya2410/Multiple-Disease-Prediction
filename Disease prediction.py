import streamlit as st
import numpy as np
import pickle
import joblib
import xgboost as xgb

# ===== Load Models =====
def load_model_kidney():
    with open("C:/Users/karunya/Documents/Guvi projects/Multiple Disease Prediction/kidney_model.sav", "rb") as f:
        model = pickle.load(f)
    features = [
        "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
        "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria", 
        "blood_glucose_random", "blood_urea", "serum_creatinine",
        "sodium", "potassium", "hemoglobin", "packed_cell_volume",
        "white_blood_cell_count", "red_blood_cell_count", 
        "hypertension", "diabetes_mellitus", "coronary_artery_disease",
        "appetite", "pedal_edema", "anemia"
    ]
    return model, features


def load_model_liver():
    model = joblib.load("C:/Users/karunya/Documents/Guvi projects/Multiple Disease Prediction/liver_disease_xgb_model.sav")
    features = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                'Albumin', 'Albumin_and_Globulin_Ratio']
    scaler = joblib.load("C:/Users/karunya/Documents/Guvi projects/Multiple Disease Prediction/scaler_liver.pkl")
    return model, features, scaler

def load_model_parkinsons():
    model = xgb.XGBClassifier()
    model.load_model("C:/Users/karunya/Documents/Guvi projects/Multiple Disease Prediction/model_parkinsons.json")
    with open("C:/Users/karunya/Documents/Guvi projects/Multiple Disease Prediction/model_parkinsons_features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("C:/Users/karunya/Documents/Guvi projects/Multiple Disease Prediction/scaler_parkinsons.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, features, scaler

# ===== Sidebar =====
st.sidebar.title("Disease Prediction System")
disease_option = st.sidebar.selectbox(
    "Select Disease for Prediction",
    ["Kidney Disease", "Liver Disease", "Parkinson's Disease"],
)

st.title("🧬 Multiple Disease Prediction App")

# ===== Form Builder =====
def build_input_form(features, special_handling={}):
    vals = []
    for feat in features:
        if feat in special_handling:
            opts = special_handling[feat]
            choice = st.selectbox(f"Select {feat}:", opts)
            vals.append(opts.index(choice))
        else:
            val = st.number_input(f"Enter {feat}:", step=0.01, format="%.2f")
            vals.append(val)
    return np.array([vals])
 
# ===== Kidney Disease Prediction =====
if disease_option == "Kidney Disease":
    model, features = load_model_kidney()
    st.subheader("🩺 Kidney Disease Input Section")
    X = build_input_form(features)
    
    if st.button("Predict Kidney Disease"):
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
        
        st.success(f"🧬 Prediction: {'At Risk' if pred==1 else 'Not at Risk'}")
        if prob is not None:
            st.info(f"📊 Probability of Risk: {prob:.2%}")

# ===== Liver Disease Prediction =====
elif disease_option == "Liver Disease":
    model, features, scaler = load_model_liver()
    st.subheader("🧪 Liver Disease Input Section")

    special_handling = {"Gender": ["Female", "Male"]} if "Gender" in features else {}

    X_raw = build_input_form(features, special_handling)

    if st.button("Predict Liver Disease"):
        if np.any(X_raw == 0):
            st.error("⚠️ Please fill all fields properly before predicting!")
        else:
            X_scaled = scaler.transform(X_raw)
            prob = model.predict_proba(X_scaled)[0][1]
            threshold = 0.5
            pred = 1 if prob > threshold else 0

            st.success(f"Prediction: {'🛑 At Risk' if pred == 1 else '✅ Not at Risk'}")
            st.info(f"Probability of Risk: {prob:.2%}")

# ===== Parkinson's Disease Prediction =====
elif disease_option == "Parkinson's Disease":
    model, features, scaler = load_model_parkinsons()
    st.subheader("🧠 Parkinson's Disease Input Section")

    X_raw = build_input_form(features)

    if st.button("Predict Parkinson's Disease"):
        if np.any(X_raw == 0):
            st.error("⚠️ Please fill all fields properly before predicting!")
        else:
            X_scaled = scaler.transform(X_raw)
            prob = model.predict_proba(X_scaled)[0][1]
            threshold = 0.6
            pred = 1 if prob > threshold else 0

            st.success(f"Prediction: {'🛑 At Risk' if pred == 1 else '✅ Not at Risk'}")
            st.info(f"Probability of Risk: {prob:.2%}")
