import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os, re

# Set page config and background image
st.set_page_config(page_title="Career Guidance App", page_icon="🎓", layout="centered")

# Add background image using custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://www.shutterstock.com/shutterstock/videos/1107672785/thumb/1.jpg?ip=x480");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- LOGIN PAGE --- #
def login_page():
    st.title("🔐 AI-BASED CAREER GUIDANCE")
    name = st.text_input("👤 Your Name")
    email = st.text_input("📧 Your Email")
    if st.button("SUBMIT"):
        if name and email:
            st.session_state["logged_in"] = True
            st.session_state["user_name"] = name
            st.session_state["user_email"] = email
            st.rerun()
        else:
            st.warning("Please fill in both your name and email.")

# Initialize session state keys
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show login page and stop if not logged in
if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# User is logged in
USER_NAME = st.session_state["user_name"]
USER_EMAIL = st.session_state["user_email"]

# Sidebar
st.sidebar.write(f"👋 Hello, {USER_NAME}")
if st.sidebar.button("🔓 Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# Utility
def format_inr(amount):
    s = str(int(amount)); last3 = s[-3:]; rest = s[:-3]
    if rest:
        rest = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", rest)
        return f"₹{rest},{last3}"
    return f"₹{last3}"

@st.cache_data
def load_data():
    try:
        return pd.read_csv("career_guidance_dataset.csv")
    except FileNotFoundError:
        st.error("❌ 'career_guidance_dataset.csv' not found.")
        st.stop()

df = load_data()

feature_cols = [
    'CGPA', 'Skills_Count', 'Internships_Count', 'Certifications_Count', 'Courses_Completed',
    'Technical_Skill_Score', 'Communication_Skill_Score', 'Projects_Count',
    'Hackathons/SalesPitch_Events', 'Publications/Blog_Count', 'Leadership_Experience',
    'Open_Source_Contributor', 'English_Proficiency', 'Professional_Platform_Rating',
    'Time_Spent_on_Learning_per_Week'
]
target_cols = ['Readiness_Score', 'Expected_Salary', 'Goal_Alignment']

X = df[feature_cols]
y = df[target_cols]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

def recommend_careers(readiness, alignment):
    if readiness > 80 and alignment > 80:
        return ["Data Scientist", "Product Manager", "AI Researcher"]
    elif readiness > 60:
        return ["Software Developer", "Business Analyst", "System Engineer"]
    else:
        return ["Trainee", "Support Executive", "Customer Success Associate"]

career_resources = {
    "Data Scientist": "https://www.coursera.org/specializations/data-science-python",
    "Software Developer": "https://www.codecademy.com/catalog/subject/computer-science",
    "Business Analyst": "https://www.edx.org/professional-certificate/business-analytics",
    "Product Manager": "https://www.udacity.com/course/product-manager-nanodegree--nd036",
    "AI Researcher": "https://www.deeplearning.ai/",
    "Trainee": "https://www.linkedin.com/learning/"
}

# Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "📚 Learning Tracker", "📌 Skill Gap Analysis", "📊 Model Performance", "📄 Dataset", "📝 Feedback"])

# Home
if page == "🏠 Home":
    st.title("🎓 Career Guidance Platform")
    st.markdown("""
    Welcome to the Career Guidance App!
    - 🎯 Predict your career readiness and salary
    - 🧠 Discover skill gaps
    - 📈 Track your learning
    - 💼 Get personalized job recommendations
    """)

# Predict
elif page == "📈 Predict":
    st.title("🎯 Career Guidance Predictor")
    st.caption("Get insights on readiness, salary, alignment & career suggestions.")
    col1, col2 = st.columns(2)
    user_input = {}
    for i, feature in enumerate(feature_cols):
        with (col1 if i % 2 == 0 else col2):
            mean_val = float(df[feature].mean())
            user_input[feature] = st.slider(f"{feature.replace('_', ' ')}", 0.0, 100.0, mean_val)
    input_df = pd.DataFrame([user_input])

    if st.button("🔮 Predict My Career Metrics"):
        prediction = model.predict(input_df)[0]
        st.success("✅ Prediction Complete!")
        salary_inr = format_inr(prediction[1])
        salary_lpa = f"₹{prediction[1]/1e5:.2f} LPA"
        col1, col2, col3 = st.columns(3)
        col1.metric("🧠 Readiness", f"{prediction[0]:.2f}/100")
        col2.metric("💰 Salary (INR)", f"{salary_inr} ({salary_lpa})")
        col3.metric("🎯 Goal Alignment", f"{prediction[2]:.2f}/100")

        st.subheader("🎓 Career Recommendations")
        careers = recommend_careers(prediction[0], prediction[2])
        for career in careers:
            st.markdown(f"- ✅ {career}")

        st.subheader("📚 Resource Links")
        for career in careers:
            if career in career_resources:
                st.markdown(f"🔗 [{career} Resource]({career_resources[career]})")

        st.subheader("📊 Prediction Summary")
        fig, ax = plt.subplots()
        bars = ax.bar(["Readiness", "Salary (L)", "Alignment"], [prediction[0], prediction[1]/100000, prediction[2]], color=["#4CAF50", "#2196F3", "#FF9800"])
        ax.set_ylabel("Score / Value")
        ax.set_title("Career Metrics")
        ax.bar_label(bars, fmt="%.2f", padding=3)
        st.pyplot(fig)

        result = pd.DataFrame({
            "Readiness_Score": [round(prediction[0], 2)],
            "Expected_Salary (INR)": [salary_inr],
            "Goal_Alignment": [round(prediction[2], 2)],
            "Recommendations": [", ".join(careers)]
        })
        st.download_button("⬇ Download Result", result.to_csv(index=False), "career_prediction.csv")

        history_path = "user_history.csv"
        if not os.path.exists(history_path):
            result.to_csv(history_path, index=False)
        else:
            result.to_csv(history_path, mode='a', header=False, index=False)

# Learning Tracker
elif page == "📚 Learning Tracker":
    st.title("📚 Weekly Learning Progress")
    week = st.selectbox("Week Number", range(1, 53))
    hours = st.slider("Hours Spent on Learning", 0, 40, 5)
    if st.button("📥 Submit Weekly Log"):
        entry = pd.DataFrame({"Week": [week], "Hours": [hours]})
        log_path = "learning_log.csv"
        if not os.path.exists(log_path):
            entry.to_csv(log_path, index=False)
        else:
            entry.to_csv(log_path, mode='a', header=False, index=False)
        st.success("✅ Logged successfully!")
    if os.path.exists("learning_log.csv"):
        log = pd.read_csv("learning_log.csv")
        st.line_chart(log.set_index("Week"))

# Skill Gap
elif page == "📌 Skill Gap Analysis":
    st.title("📌 Skill Gap Analyzer")
    ideal_profile = df[feature_cols].mean()
    col1, col2 = st.columns(2)
    user_input = {}
    for i, feature in enumerate(feature_cols):
        with (col1 if i % 2 == 0 else col2):
            user_input[feature] = st.slider(f"{feature.replace('_', ' ')}", 0.0, 100.0, ideal_profile[feature])
    user_df = pd.DataFrame([user_input])
    gap = pd.DataFrame({
        "Feature": feature_cols,
       

        "Your Score": user_df.iloc[0],
        "Ideal Score": ideal,
        "Gap": ideal - user_df.iloc[0]
    }).sort_values("Gap", ascending=False)
    st.subheader("📉 Areas for Improvement")
    st.dataframe(gap[gap["Gap"]>0].head(10))

elif page == "📊 Model Performance":
    st.title("📊 Model Evaluation on Test Data")
    for idx, col in enumerate(target_cols):
        st.subheader(f"{col}")
        st.write(f"🔻 Mean Squared Error: {mean_squared_error(y_test[col], y_pred[:, idx]):.2f}")
        st.write(f"📈 R² Score: {r2_score(y_test[col], y_pred[:, idx]):.2f}")

elif page == "📄 Dataset":
    st.title("📄 Sample Dataset and History")
    st.dataframe(df.head(20))
    hist = "user_history.csv"
    if os.path.exists(hist):
        st.subheader("🗂 Your Previous Predictions")
        st.dataframe(pd.read_csv(hist).tail(10))
    else:
        st.info("No user history available yet.")

elif page == "📝 Feedback":
    st.title("📝 We Value Your Feedback")
    fname = st.text_input("Your Name")
    rating = st.slider("How helpful was this tool?", 1, 5, 4)
    comment = st.text_area("Any suggestions or comments?")
    if st.button("📩 Submit Feedback"):
        fb = pd.DataFrame({"Name":[fname],"Rating":[rating],"Comment":[comment]})
        path = "feedback.csv"
        fb.to_csv(path, index=False, mode='a', header=not os.path.exists(path))
        st.success("✅ Feedback submitted. Thank you!")