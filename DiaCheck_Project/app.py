from pathlib import Path
from datetime import datetime
from io import BytesIO
import joblib
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "diabetes_risk_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.joblib"

st.set_page_config(page_title="DiaCheck | Educational Assessment", page_icon="🩺")
st.title("🩺 DiaCheck")
st.subheader("Diabetes risk-factor educational assessment")
st.warning("Coursework prototype only — NOT a medical device or diagnostic tool. The model may be wrong. Do not use this output to diagnose, rule out, or treat diabetes. Consult a qualified clinician for health concerns.")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Model not trained yet. Follow README.md: download the documented dataset into data/ and run train_model.py.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)
st.write("Enter general health and lifestyle factors. This version does not ask for lab values or symptoms. Use fictional/demo values for coursework.")

with st.form("risk_form"):
    st.markdown("### General risk factors")
    age = st.selectbox("Age group", list(range(1, 14)), index=5,
        format_func=lambda x: {1:"18–24",2:"25–29",3:"30–34",4:"35–39",5:"40–44",6:"45–49",7:"50–54",8:"55–59",9:"60–64",10:"65–69",11:"70–74",12:"75–79",13:"80 or older"}[x])
    bmi = st.number_input("BMI (body mass index)", 10.0, 80.0, 25.0, 0.5)
    high_bp = st.selectbox("Ever told you have high blood pressure?", ["No", "Yes"])
    high_chol = st.selectbox("Ever told you have high cholesterol?", ["No", "Yes"])
    smoker = st.selectbox("Have you smoked at least 100 cigarettes in your life?", ["No", "Yes"])
    activity = st.selectbox("Any physical activity/exercise in past 30 days (outside regular job)?", ["Yes", "No"])
    fruits = st.selectbox("Consume fruit at least once per day?", ["Yes", "No"])
    veggies = st.selectbox("Consume vegetables at least once per day?", ["Yes", "No"])
    health = st.select_slider("Self-reported general health", options=[1,2,3,4,5], value=3,
        format_func=lambda x:{1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"}[x])
    walk = st.selectbox("Serious difficulty walking or climbing stairs?", ["No", "Yes"])
    sex = st.selectbox("Sex category used in source survey", ["Female", "Male"])
    education = st.selectbox("Education category (source survey scale)", list(range(1,7)), index=4,
        format_func=lambda x:{1:"Never attended/kindergarten",2:"Grades 1–8",3:"Grades 9–11",4:"High school graduate",5:"Some college/technical school",6:"College graduate"}[x])
    income = st.selectbox("Household income category (source survey scale)", list(range(1,9)), index=4,
        format_func=lambda x:{1:"Under $10,000",2:"$10,000–$15,000",3:"$15,000–$20,000",4:"$20,000–$25,000",5:"$25,000–$35,000",6:"$35,000–$50,000",7:"$50,000–$75,000",8:"$75,000 or more"}[x])
    submitted = st.form_submit_button("Estimate educational risk pattern")

if submitted:
    raw = {
        "HighBP": int(high_bp == "Yes"), "HighChol": int(high_chol == "Yes"),
        "BMI": float(bmi), "Smoker": int(smoker == "Yes"),
        "PhysActivity": int(activity == "Yes"), "Fruits": int(fruits == "Yes"),
        "Veggies": int(veggies == "Yes"), "GenHlth": int(health),
        "DiffWalk": int(walk == "Yes"), "Sex": int(sex == "Male"),
        "Age": int(age), "Education": int(education), "Income": int(income)
    }
    row = pd.DataFrame([{col: raw.get(col, 0) for col in feature_columns}], columns=feature_columns)
    score = float(model.predict_proba(row)[0][1])
    st.markdown("### Result")
    st.metric("Model score for the dataset's positive class", f"{score:.1%}")
    st.caption("This is not a personal medical probability. It is not calibrated or validated for clinical use.")
    st.info("The source target is based on self-reported survey categories, not necessarily a clinician-confirmed diagnosis. This result cannot confirm or exclude diabetes. A clinician can advise whether testing is appropriate.")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 17)
    c.drawString(45, y, "DiaCheck — Educational Risk-Factor Report")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(45, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(45, y, "Entered factors")
    y -= 18
    labels = {"HighBP":"High blood pressure","HighChol":"High cholesterol","BMI":"BMI","Smoker":"Smoking history","PhysActivity":"Physical activity","Fruits":"Daily fruit","Veggies":"Daily vegetables","GenHlth":"General health (1=excellent, 5=poor)","DiffWalk":"Difficulty walking","Sex":"Survey sex category (0=female, 1=male)","Age":"Age group (survey code)","Education":"Education category","Income":"Income category"}
    c.setFont("Helvetica", 9)
    for key in feature_columns:
        c.drawString(45, y, f"{labels.get(key,key)}: {raw.get(key, 'Not entered')}")
        y -= 14
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(45, y, f"Educational model score: {score:.1%}")
    y -= 22
    c.setFont("Helvetica", 9)
    disclaimer = ("IMPORTANT: Coursework only. Not a diagnosis, medical advice, or substitute for a clinician. "
                  "Trained on a US survey-derived dataset; it may not apply to other populations, including India. "
                  "Do not use this score to make health decisions. Consult a qualified healthcare professional.")
    for start in range(0, len(disclaimer), 100):
        c.drawString(45, y, disclaimer[start:start+100])
        y -= 13
    c.save()
    st.download_button("Download PDF report", buffer.getvalue(),
                       file_name="diacheck_educational_report.pdf", mime="application/pdf")

with st.expander("Dataset and limitations"):
    st.write("This learning project uses a public CDC Behavioral Risk Factor Surveillance System (BRFSS)-derived Diabetes Health Indicators dataset. It is US-focused and self-reported. Its target is survey-based, not necessarily a clinical diagnosis. Results may not generalize to India or any individual. Real-world use would require expert review, external validation, calibration, fairness assessment, and clinical/regulatory oversight.")
