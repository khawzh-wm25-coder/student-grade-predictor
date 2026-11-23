import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Student Grade Early-Warning System", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Early-Warning Prototype")
st.markdown(
    "Predict a student's final grade (**G3**, scale 0–20) using the UCI Student Performance dataset "
    "to help educators identify at-risk students early."
)

# 1. Load data and train model (cached so it runs fast)
@st.cache_resource
def load_and_train():
    # UCI dataset uses semicolons as delimiters
    df = pd.read_csv("student-mat.csv", sep=";")
    
    features = ["G1", "G2", "failures", "absences", "studytime"]
    target = "G3"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return model, rmse, r2

model, rmse, r2 = load_and_train()

# 2. Sidebar inputs for teachers/advisors
st.sidebar.header("📋 Enter Student Profile")

g1 = st.sidebar.slider("First Period Grade (G1)", min_value=0, max_value=20, value=11)
g2 = st.sidebar.slider("Second Period Grade (G2)", min_value=0, max_value=20, value=11)
failures = st.sidebar.selectbox("Past Class Failures", options=[0, 1, 2, 3], index=0)
absences = st.sidebar.slider("School Absences (Days)", min_value=0, max_value=75, value=4)
studytime = st.sidebar.selectbox(
    "Weekly Study Time",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 (< 2 hours)",
        2: "2 (2 to 5 hours)",
        3: "3 (5 to 10 hours)",
        4: "4 (> 10 hours)"
    }[x],
    index=1
)

# 3. Generate Prediction
input_data = pd.DataFrame([[g1, g2, failures, absences, studytime]],
                          columns=["G1", "G2", "failures", "absences", "studytime"])
predicted_g3 = model.predict(input_data)[0]

# 4. Display Results
st.subheader("📊 Prediction Output")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted Final Grade (G3)", value=f"{predicted_g3:.1f} / 20")

with col2:
    if predicted_g3 < 10:
        st.error("🚨 High Risk (Projected Fail: G3 < 10)")
    elif predicted_g3 < 13:
        st.warning("⚠️ Moderate Risk (Borderline Pass)")
    else:
        st.success("✅ Low Risk (On Track)")

# Intervention Guidance
st.markdown("### 💡 Recommended Action")
if predicted_g3 < 10:
    st.write("Immediate academic advisor intervention recommended: schedule tutoring and review attendance records.")
elif absences > 10 or failures > 0:
    st.write("Monitor attendance and assign targeted study-skills support.")
else:
    st.write("Student is performing well; continue standard progress tracking.")

# Model Performance Footer (links directly to Task 1 Success Criteria)
with st.expander("ℹ️ Model Validation Metrics (Test Set)"):
    st.write(f"**Algorithm:** Random Forest Regressor")
    st.write(f"**Test RMSE:** {rmse:.2f} grade points (Target: < 2.0)")
    st.write(f"**Test R²:** {r2:.2f} (Target: > 0.80)")
