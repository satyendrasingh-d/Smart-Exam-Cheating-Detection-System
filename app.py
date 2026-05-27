
# ======================================================
# Smart Exam Cheating Detection System
# Streamlit Application
# ======================================================

# Import required libraries
import streamlit as st
import joblib
import pandas as pd

# ======================================================
# LOAD TRAINED MODEL
# ======================================================

# Load saved ML model
model = joblib.load('cheating_model.pkl')

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Smart Exam Cheating Detection",
    page_icon="🛡️",
    layout="centered"
)

# ======================================================
# PAGE TITLE
# ======================================================

st.title("🛡️ Smart Exam Cheating Detection System")

st.write(
    "Machine Learning based system to detect suspicious "
    "behavior during online examinations."
)

# ======================================================
# SIDEBAR INFORMATION
# ======================================================

st.sidebar.header("📌 About Project")

st.sidebar.write("""
This project uses Machine Learning to analyze:

- Tab switching
- Typing speed
- Idle time
- Mouse movement
- Copy-paste activity

and predicts whether the behavior is:

✅ Normal  
❌ Suspicious
""")

# ======================================================
# USER INPUT SECTION
# ======================================================

st.subheader("📊 Enter Student Activity Data")

# Tab switching
tab_switches = st.slider(
    "Number of Tab Switches",
    min_value=0,
    max_value=15,
    value=2
)

# Typing speed
typing_speed = st.slider(
    "Typing Speed (Words Per Minute)",
    min_value=0,
    max_value=100,
    value=40
)

# Idle time
idle_time = st.slider(
    "Idle Time (Seconds)",
    min_value=0,
    max_value=120,
    value=10
)

# Mouse movement
mouse_movements = st.slider(
    "Mouse Movement Count",
    min_value=0,
    max_value=300,
    value=100
)

# Copy paste count
copy_paste_count = st.slider(
    "Copy Paste Count",
    min_value=0,
    max_value=15,
    value=1
)

# ======================================================
# PREDICTION BUTTON
# ======================================================

if st.button("🔍 Analyze Behavior"):

    # Create dataframe from user inputs
    input_data = pd.DataFrame({

        'tab_switches': [tab_switches],

        'typing_speed': [typing_speed],

        'idle_time': [idle_time],

        'mouse_movements': [mouse_movements],

        'copy_paste_count': [copy_paste_count]

    })

    # ==================================================
    # MODEL PREDICTION
    # ==================================================
    # Predict behavior
    prediction = model.predict(input_data)[0]

    # Prediction probability
    probability = model.predict_proba(input_data)

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    st.subheader("📢 Prediction Result")

    # Normal behavior
    if prediction == 0:

        confidence = round(
            probability[0][0] * 100,
            2
        )

        st.success(
            f"✅ Normal Exam Behavior Detected"
        )

        st.write(
            f"Confidence Score: {confidence}%"
        )

    # Suspicious behavior
    else:

        risk_score = round(
            probability[0][1] * 100,
            2
        )

        st.error(
            f"❌ Suspicious Exam Behavior Detected"
        )

        st.write(
            f"Risk Score: {risk_score}%"
        )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
    "Developed using Machine Learning and Streamlit"
)
