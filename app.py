import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Grade Early-Warning System", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Early-Warning Prototype")
st.markdown(
    "Predict final grades (**G3**, scale 0–20) using the UCI Student Performance dataset "
    "to identify at-risk students early."
)

FEATURES = ["G1", "G2", "failures", "absences", "studytime"]

@st.cache_resource
def load_and_train():
    df = pd.read_csv("student-mat.csv", sep=";")
    X = df[FEATURES]
    y = df["G3"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, np.sqrt(mean_squared_error(y_test, preds)), r2_score(y_test, preds)

model, rmse, r2 = load_and_train()

# --- Sidebar: Single Student Prediction ---
st.sidebar.header("📋 Single Student Profile")
g1 = st.sidebar.slider("First Period Grade (G1)", 0, 20, 11)
g2 = st.sidebar.slider("Second Period Grade (G2)", 0, 20, 11)
failures = st.sidebar.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
absences = st.sidebar.slider("School Absences (Days)", 0, 75, 4)
studytime = st.sidebar.selectbox(
    "Weekly Study Time", [1, 2, 3, 4],
    format_func=lambda x: {1: "1 (< 2 hrs)", 2: "2 (2–5 hrs)", 3: "3 (5–10 hrs)", 4: "4 (> 10 hrs)"}[x],
    index=1
)

# --- Tab Layout: Single vs. Batch Prediction ---
tab1, tab2 = st.tabs(["👤 Single Student Prediction", "📂 Batch Class CSV Upload"])

with tab1:
    input_df = pd.DataFrame([[g1, g2, failures, absences, studytime]], columns=FEATURES)
    pred_g3 = model.predict(input_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Final Grade (G3)", f"{pred_g3:.1f} / 20")
    with col2:
        if pred_g3 < 10:
            st.error("🚨 High Risk (Projected Fail: G3 < 10)")
        elif pred_g3 < 13:
            st.warning("⚠️ Moderate Risk (Borderline Pass)")
        else:
            st.success("✅ Low Risk (On Track)")

with tab2:
    st.subheader("Upload Class Roster CSV")
    st.write(f"Upload a CSV containing columns: `{', '.join(FEATURES)}` (comma- or semicolon-separated).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Auto-detect semicolon vs comma delimiter
        raw_df = pd.read_csv(uploaded_file, sep=None, engine="python")
        if all(col in raw_df.columns for col in FEATURES):
            results_df = raw_df.copy()
            results_df["Predicted_G3"] = np.round(model.predict(raw_df[FEATURES]), 2)
            results_df["Risk_Level"] = np.where(
                results_df["Predicted_G3"] < 10, "High Risk",
                np.where(results_df["Predicted_G3"] < 13, "Moderate Risk", "Low Risk")
            )
            st.dataframe(results_df[FEATURES + ["Predicted_G3", "Risk_Level"]], use_container_width=True)
            
            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="predicted_student_grades.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Missing required columns. Ensure your file contains: {FEATURES}")

with st.expander("ℹ️ Model Validation Metrics (Test Set)"):
    st.write(f"**Algorithm:** Random Forest Regressor | **Test RMSE:** {rmse:.2f} | **Test R²:** {r2:.2f}")
