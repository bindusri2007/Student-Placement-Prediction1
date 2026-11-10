
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("placement_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# Application title
st.title("🎓 Student Placement Eligibility Prediction")

st.write(
    "Enter your academic details and skills "
    "to predict your placement status."
)

st.divider()

# Student details
st.header("Student Academic Details")

cgpa = st.number_input(
    "B.Tech CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

tenth = st.number_input(
    "10th Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

intermediate = st.number_input(
    "Intermediate / Diploma Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

st.header("Skills Information")

technical = st.slider(
    "Technical Skills (1-10)",
    1, 10, 7
)

communication = st.slider(
    "Communication Skills (1-10)",
    1, 10, 7
)

internship = st.selectbox(
    "Have you completed an internship?",
    ["Yes", "No"]
)

backlogs = st.number_input(
    "Number of Backlogs",
    min_value=0,
    max_value=20,
    value=0
)

# Prediction button
if st.button("Predict Placement Eligibility"):

    # Prepare student input
    student = pd.DataFrame([{
        'CGPA': cgpa,
        '10th_Percentage': tenth,
        'Intermediate_Diploma_Percentage': intermediate,
        'Technical_Skills_Score': technical,
        'Communication_Skills_Score': communication,
        'Backlogs': backlogs,
        'Internship': 1 if internship == "Yes" else 0
    }])

    # Make prediction
    prediction = model.predict(student)[0]

    probability = model.predict_proba(student)[0]

    st.divider()
    st.header("Prediction Result")

    if prediction == 1:
        st.success("Eligible for Placement")
    else:
        st.error("Not Eligible for Placement")

    st.write(
        "Placed Probability:",
        round(probability[1] * 100, 2), "%"
    )

    st.write(
        "Not Placed Probability:",
        round(probability[0] * 100, 2), "%"
    )

    st.info(
        "This is an educational ML prediction, "
        "not an official company eligibility decision."
    )
