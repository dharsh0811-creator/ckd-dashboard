import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CKD Risk Screening Dashboard", page_icon="🩺", layout="wide"
)

st.title("🩺 Smart Wearable-Based CKD Risk Screening Dashboard")
st.caption("Model: Random Forest (80% Test Accuracy)")

MODEL_PATH = "CKD_Random_Forest_Model.pkl"


@st.cache_resource
def load_model():
  return joblib.load(MODEL_PATH)


try:
  model = load_model()
except Exception as e:
  st.error(
      f"'{MODEL_PATH}' kaanala. Rendu file-um ore folder-la irukkaanu check"
      f" pannunga.\nError: {e}"
  )
  st.stop()

st.sidebar.header("📊 Smartwatch Biometric Inputs")
mean_hr = st.sidebar.slider("Mean HR (bpm)", 45.0, 130.0, 76.0, 0.5)
min_hr = st.sidebar.slider("Min HR (bpm)", 35.0, 100.0, 56.0, 0.5)
max_hr = st.sidebar.slider("Max HR (bpm)", 70.0, 180.0, 105.0, 0.5)
hr_std = st.sidebar.slider("HR Std (bpm)", 1.0, 30.0, 8.5, 0.1)
mean_spo2 = st.sidebar.slider("Mean SpO2 (%)", 85.0, 100.0, 97.5, 0.1)
sleep_hours = st.sidebar.slider("Sleep Hours (hrs)", 2.0, 12.0, 7.0, 0.1)
daily_steps = st.sidebar.number_input(
    "Daily Steps", min_value=0, max_value=30000, value=7500, step=100
)

input_df = pd.DataFrame([{
    "Mean_HR": mean_hr,
    "Min_HR": min_hr,
    "Max_HR": max_hr,
    "HR_Std": hr_std,
    "Mean_SpO2": mean_spo2,
    "Sleep_Hours": sleep_hours,
    "Daily_Steps": daily_steps,
}])

col1, col2 = st.columns([1, 1.2])

with col1:
  st.subheader("🎯 Risk Prediction")
  if st.button("Predict CKD Risk Tier", type="primary"):
    pred = int(model.predict(input_df)[0])
    probs = model.predict_proba(input_df)[0]

    risk_map = {
        0: ("🟢 Low Risk (Tier 0)", "Biometric profile stable.", "success"),
        1: (
            "🟡 Medium Risk (Tier 1)",
            "Moderate risk — clinical follow-up advised.",
            "warning",
        ),
        2: (
            "🔴 High Risk (Tier 2)",
            "Elevated risk — medical evaluation recommended.",
            "error",
        ),
    }
    title_txt, desc_txt, status_type = risk_map.get(
        pred, (f"Tier {pred}", "Predicted risk tier.", "info")
    )

    if status_type == "success":
      st.success(f"### {title_txt}\n{desc_txt}")
    elif status_type == "warning":
      st.warning(f"### {title_txt}\n{desc_txt}")
    else:
      st.error(f"### {title_txt}\n{desc_txt}")

    prob_df = pd.DataFrame({
        "Risk Tier": [f"Tier {c}" for c in model.classes_],
        "Probability (%)": np.round(probs * 100, 1),
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

with col2:
  st.subheader("📌 Grouped Feature Importance")
  feature_cols = [
      "Mean_HR",
      "Min_HR",
      "Max_HR",
      "HR_Std",
      "Mean_SpO2",
      "Sleep_Hours",
      "Daily_Steps",
  ]
  raw_imp = pd.Series(model.feature_importances_, index=feature_cols)

  grouped_imp = pd.Series({
      "Heart Rate (Combined)": raw_imp[
          ["Mean_HR", "Min_HR", "Max_HR", "HR_Std"]
      ].sum(),
      "Daily Steps": raw_imp["Daily_Steps"],
      "Mean SpO2": raw_imp["Mean_SpO2"],
      "Sleep Hours": raw_imp["Sleep_Hours"],
  }).sort_values(ascending=True)

  fig, ax = plt.subplots(figsize=(6.5, 4))
  grouped_imp.plot(kind="barh", ax=ax, color="#2b6cb0")
  ax.set_xlabel("Relative Importance")
  plt.tight_layout()
  st.pyplot(fig)
