import streamlit as st
import pandas as pd
import joblib

# Load models
placement_classifier = joblib.load("placement_classifier.pkl")
salary_regressor = joblib.load("salary_regressor.pkl")

# Page config
st.set_page_config(page_title="Student Placement and Salary Prediction System", page_icon="🎓", layout="wide")

# Custom CSS styles
st.markdown(
    """
    <style>
    /* Background gradient */
    .main {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        padding: 2rem 5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Header style */
    h1 {
        color: #0b3d91;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px #a3c1ff;
    }
    /* Button style */
    div.stButton > button:first-child {
        background-color: #0b3d91;
        color: white;
        font-weight: bold;
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1e69d2;
        color: #e0e7ff;
    }
    /* Result box */
    .result-box {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.15);
        margin-top: 2rem;
    }
    /* Subheader style */
    h3 {
        color: #0b3d91;
        margin-bottom: 1rem;
    }
    /* Input label colors */
    label {
        color: #1e69d2 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown("<h1>🎓 Student Placement & Salary Prediction System</h1>", unsafe_allow_html=True)
st.write(
    "<p style='text-align:center; color:#0b3d91; font-size:1.1rem;'>Fill in your details below to get your placement status and expected salary prediction.</p>",
    unsafe_allow_html=True
)

# Inputs spliting equally: 12 attributes / 3 columns = 4 per column
col1, col2, col3 = st.columns(3)

with col1:
    cgpa = st.number_input("CGPA (5.0 - 10.0)", min_value=5.0, max_value=10.0, step=0.01)
    tenth_percentage = st.number_input("10th Percentage (50 - 100)", min_value=50.0, max_value=100.0, step=0.01)
    twelfth_percentage = st.number_input("12th Percentage (50 - 100)", min_value=50.0, max_value=100.0, step=0.01)
    internships = st.number_input("Internships Completed", min_value=0, max_value=10, step=1)

with col2:
    certifications = st.number_input("Certifications Earned", min_value=0, max_value=10, step=1)
    communication = st.number_input("Communication Skills (1-10)", min_value=1, max_value=10, step=1)
    technical = st.number_input("Technical Skills (1-10)", min_value=1, max_value=10, step=1)
    aptitude = st.number_input("Aptitude Test Score (0-100)", min_value=0.0, max_value=100.0, step=0.01)

with col3:
    projects = st.number_input("Projects Completed", min_value=0, max_value=10, step=1)
    hackathons = st.number_input("Hackathons Participated", min_value=0, max_value=10, step=1)
    problem_solving = st.number_input("Problem Solving Skills (1-10)", min_value=1, max_value=10, step=1)
    interview_score = st.number_input("Interview Score (1-10)", min_value=1, max_value=10, step=1)

if st.button("🔍 Predict Placement & Salary"):
    input_data = pd.DataFrame([[
        cgpa, tenth_percentage, twelfth_percentage, internships,
        certifications, communication, technical, aptitude,
        projects, hackathons, problem_solving, interview_score
    ]], columns=[
        "CGPA", "10th_Percentage", "12th_Percentage", "Internships",
        "Certifications", "Communication_Skill", "Technical_Skill", "Aptitude_Score",
        "Projects", "Hackathons", "Problem_Solving", "Interview_Score"
    ])

    placement_prediction = placement_classifier.predict(input_data)[0]
    salary_prediction = salary_regressor.predict(input_data)[0]

    placement_mapping = {0: "Not Placed", 1: "Placed", 2: "Dream Offer"}
    placement_status = placement_mapping.get(placement_prediction, "Unknown")

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Results")
    st.markdown(f"**Placement Status:** <span style='color:#0b3d91; font-weight:bold;'>{placement_status}</span>", unsafe_allow_html=True)

    if placement_status in ["Placed", "Dream Offer"]:
        st.markdown(f"**Expected Monthly Salary:** <span style='color:#1e69d2; font-weight:bold;'>₹{salary_prediction:,.2f}</span>", unsafe_allow_html=True)
    else:
        st.markdown("**Expected Monthly Salary:** <span style='color:#b03030; font-weight:bold;'>₹0.00</span>", unsafe_allow_html=True)

    if placement_status == "Placed":
        st.success("🎉 Congratulations! You have good chances of getting placed.")
    elif placement_status == "Dream Offer":
        st.balloons()
        st.success("🚀 Wow! You are on track for a dream offer!")
    else:
        st.error("📌 You might need to improve your profile for better placement chances.")
    st.markdown("</div>", unsafe_allow_html=True)
