"""
Student Mental Health Score Predictor — Streamlit Frontend
------------------------------------------------------------
This app is the UI layer for the FastAPI service defined in `main.py`.
It collects the exact fields required by the `StudentData` Pydantic model,
sends them to the `/predict` endpoint, and displays the predicted
Mental_Health_Score in a friendly, visual way.

Run the API first:
    uvicorn main:app --reload --port 8000

Then run this app:
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .stApp { background-color: #0f1116; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1rem; }
    .result-card {
        padding: 1.8rem;
        border-radius: 16px;
        background: #1a1d29;
        border: 1px solid #2d3142;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.7rem 0;
        font-weight: 600;
        font-size: 1.05rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover { opacity: 0.9; }
    section[data-testid="stSidebar"] { background-color: #14161f; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# API endpoint — set this to your deployed FastAPI URL
# --------------------------------------------------------------------------
API_URL = os.environ.get("MENTAL_HEALTH_API_URL", "https://mental-health-score-1q82.onrender.com")

# --------------------------------------------------------------------------
# Constants — mirrors the Pydantic model / dataset in main.py
# --------------------------------------------------------------------------
ALL_COUNTRIES = sorted([
    'Afghanistan', 'Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 'Austria',
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bhutan',
    'Bolivia', 'Bosnia', 'Brazil', 'Bulgaria', 'Canada', 'Chile', 'China', 'Colombia',
    'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt',
    'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Hong Kong',
    'Hungary', 'Iceland', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Israel', 'Italy',
    'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan',
    'Latvia', 'Lebanon', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malaysia', 'Maldives',
    'Malta', 'Mexico', 'Moldova', 'Monaco', 'Montenegro', 'Morocco', 'Nepal', 'Netherlands',
    'New Zealand', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama',
    'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia',
    'San Marino', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa',
    'South Korea', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Syria', 'Taiwan',
    'Tajikistan', 'Thailand', 'Trinidad', 'Turkey', 'UAE', 'UK', 'USA', 'Ukraine',
    'Uruguay', 'Uzbekistan', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Other'
])

PLATFORMS = ['Instagram', 'TikTok', 'Facebook', 'YouTube', 'Twitter', 'Snapchat',
             'LinkedIn', 'WhatsApp', 'WeChat', 'LINE', 'KakaoTalk', 'VKontakte']

PURPOSES = ['Entertainment', 'Education', 'Networking', 'News']

STRESS_LEVELS = ['Low', 'Medium', 'High', 'Very High']

STRESS_COLORS = {'Low': '#22c55e', 'Medium': '#eab308', 'High': '#f97316', 'Very High': '#ef4444'}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
api_url = API_URL.rstrip("/")

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app collects student lifestyle & social-media habits and sends "
        "them to a FastAPI service (`main.py`) which runs them through a "
        "trained ML pipeline (`Mental_Health_Model.pkl`) to estimate a "
        "**Mental Health Score (0–10)**."
    )
    st.markdown("---")
    st.caption("Built with Streamlit • Backend: FastAPI + scikit-learn")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🧠 Student Mental Health Score Predictor</h1>
    <p>Estimate a student's mental health score from their social media habits, study routine, and lifestyle.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    with st.form("prediction_form"):
        st.markdown("#### 👤 Profile")
        p1, p2, p3 = st.columns(3)
        with p1:
            age = st.number_input("Age", min_value=10, max_value=100, value=21, step=1)
        with p2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with p3:
            academic_level = st.selectbox("Academic Level", ["Undergraduate", "Graduate", "High School"])

        country = st.selectbox("Country", ALL_COUNTRIES,
                                index=ALL_COUNTRIES.index("USA") if "USA" in ALL_COUNTRIES else 0)

        st.markdown("#### 📱 Social Media Habits")
        s1, s2 = st.columns(2)
        with s1:
            most_used_platform = st.selectbox("Most Used Platform", PLATFORMS)
        with s2:
            purpose_of_use = st.selectbox("Primary Purpose of Use", PURPOSES)

        s3, s4 = st.columns(2)
        with s3:
            avg_daily_usage_hours = st.slider("Avg. Daily Usage (hours)", 0.0, 24.0, 5.0, 0.1)
        with s4:
            daily_unlocks = st.number_input("Daily Phone Unlocks", min_value=0, value=170, step=1)

        st.markdown("#### 🗓️ Daily Routine")
        r1, r2, r3 = st.columns(3)
        with r1:
            study_hours = st.slider("Study Hours/day", 0.0, 24.0, 3.0, 0.1)
        with r2:
            physical_activity_hours = st.slider("Physical Activity (hrs/day)", 0.0, 24.0, 1.8, 0.1)
        with r3:
            sleep_hours_per_night = st.slider("Sleep (hrs/night)", 0.0, 24.0, 6.6, 0.1)

        st.markdown("#### 😰 Stress")
        stress_level = st.select_slider("Current Stress Level", options=STRESS_LEVELS, value="Medium")

        submitted = st.form_submit_button("🔮 Predict Mental Health Score")

# --------------------------------------------------------------------------
# Prediction + result display
# --------------------------------------------------------------------------
with col_right:
    st.markdown("#### 📊 Result")

    if submitted:
        payload = {
            "age": int(age),
            "gender": gender,
            "country": country,
            "academic_level": academic_level,
            "most_used_platform": most_used_platform,
            "purpose_of_use": purpose_of_use,
            "avg_daily_usage_hours": float(avg_daily_usage_hours),
            "daily_unlocks": int(daily_unlocks),
            "study_hours": float(study_hours),
            "physical_activity_hours": float(physical_activity_hours),
            "sleep_hours_per_night": float(sleep_hours_per_night),
            "stress_level": stress_level,
        }

        try:
            with st.spinner("Contacting model..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=15)

            if response.status_code == 200:
                score = response.json()["predicted_mental_health_score"]
                score_clamped = max(0, min(10, score))

                if score >= 7.5:
                    label, color = "Thriving", "#22c55e"
                elif score >= 6:
                    label, color = "Doing Well", "#84cc16"
                elif score >= 4.5:
                    label, color = "Needs Attention", "#eab308"
                else:
                    label, color = "At Risk", "#ef4444"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_clamped,
                    number={'suffix': " / 10", 'font': {'size': 40, 'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 10], 'tickcolor': 'white'},
                        'bar': {'color': color},
                        'bgcolor': "#1a1d29",
                        'steps': [
                            {'range': [0, 4.5], 'color': '#3f1d1d'},
                            {'range': [4.5, 6], 'color': '#3f3a1d'},
                            {'range': [6, 7.5], 'color': '#2e3f1d'},
                            {'range': [7.5, 10], 'color': '#1d3f26'},
                        ],
                    },
                ))
                fig.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div class="result-card">
                    <span class="badge" style="background:{color}22; color:{color}; border:1px solid {color};">
                        {label}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 💡 Quick take")
                notes = []
                if sleep_hours_per_night < 6:
                    notes.append("Sleep is on the low side — under 6 hrs/night is linked to lower scores.")
                if avg_daily_usage_hours > 6:
                    notes.append("Daily social media usage is high (6+ hrs).")
                if stress_level in ("High", "Very High"):
                    notes.append("Reported stress level is elevated.")
                if physical_activity_hours < 1:
                    notes.append("Physical activity is quite low.")
                if not notes:
                    notes.append("Habits look fairly balanced overall.")
                for n in notes:
                    st.write(f"• {n}")

            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not reach the prediction service right now. Please try again shortly.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
    else:
        st.info("Fill in the form and click **Predict Mental Health Score** to see the result here.")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ This tool provides an estimate based on a machine learning model trained on survey data. "
    "It is not a clinical diagnosis. If you're struggling, please reach out to a mental health professional."
)
