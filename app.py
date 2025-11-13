import streamlit as st
import pandas as pd
import os
import base64
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- SETUP ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

POLICY_DIR = "policies"
os.makedirs(POLICY_DIR, exist_ok=True)

QUERY_FILE = "queries.csv"
if not os.path.exists(QUERY_FILE):
    pd.DataFrame(columns=["timestamp", "context", "question", "answer"]).to_csv(QUERY_FILE, index=False)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Avelora Consulting - Policy Insights", page_icon="💼", layout="wide")

# ---------- CSS (PASTEL + CUSTOM SIDEBAR + NO WHITE BOXES) ----------
st.markdown("""
<style>

/* Remove white background from sidebar content */
[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #E6E6FA, #FFF8E1) !important;
    padding-top: 25px !important;
    border-right: 1px solid #e0e0e0;
}

/* Remove default Streamlit radio style */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    background: transparent !important;
    border: none !important;
}
section[data-testid="stSidebar"] input[type="radio"] {
    display: none !important;
}
section[data-testid="stSidebar"] label {
    background: transparent !important;
}

/* Custom Menu Item */
.menu-item {
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    cursor: pointer;
    display: block;
    font-size: 16px;
    font-weight: 500;
    color: black;
    background: rgba(255,255,255,0.45);
    text-decoration: none;
    transition: 0.2s ease;
}
.menu-item:hover {
    background: rgba(255,255,255,0.7);
    transform: translateX(5px);
}
.menu-active {
    background: white !important;
    border-left: 4px solid #C39BD3;
    font-weight: 600 !important;
}

/* Header */
.header {
    background: linear-gradient(90deg, #E6E6FA, #FFF8E1);
    padding: 20px;
    border-radius: 10px;
    color: black;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #e8e8e8;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Chat bubbles */
.chat-bubble-user {
    background-color: #E8DAEF;
    padding: 12px;
    border-radius: 10px;
    margin: 10px 0;
}
.chat-bubble-bot {
    background-color: #F3E5F5;
    padding: 12px;
    border-radius: 10px;
    border-left: 4px solid #C39BD3;
    margin: 10px 0;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #F3E5F5, #E6E6FA);
    border: 1px solid #C39BD3;
    border-radius: 8px;
    color: black;
    padding: 8px 20px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #E6E6FA, #F3E5F5);
}

/* Body */
body, [data-testid="stAppViewContainer"] {
    background-color: #FAFAFA;
    font-family: "Segoe UI", sans-serif;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class='header'>
    <h1>💼 Avelora Consulting — Policy Insights Dashboard</h1>
    <p>Helping employees understand company policies with ease.</p>
</div>
""", unsafe_allow_html=True)

# ---------- HELPER FUNCTIONS ----------
def ask_ai(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a friendly HR assistant who explains policies clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

def save_query(context, question, answer):
    df = pd.read_csv(QUERY_FILE)
    new_row = pd.DataFrame([[datetime.now(), context, question, answer]],
                           columns=["timestamp","context","question","answer"])
    df = pd.concat([df,new_row], ignore_index=True)
    df.to_csv(QUERY_FILE, index=False)

def show_policy_card(file_path):
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        dl = f"<a href='data:application/octet-stream;base64,{b64}' download='{file_name}' style='color:#7D3C98;'>📥 Download</a>"
    st.markdown(f"<div class='card'><b>📄 {file_name}</b><br>{dl}</div>", unsafe_allow_html=True)

# ---------- CUSTOM SIDEBAR NAVIGATION ----------
st.sidebar.markdown("### 🧭 Navigation")

page_list = ["Home", "All Policies", "Upload or Choose & Ask", "Ask Policy AI", "My Analytics", "My FAQs"]

page = st.sidebar.radio("", page_list, label_visibility="collapsed")

# ---------- HOME PAGE ----------
if page == "Home":
    st.title("🏠 Welcome to Avelora Consulting's Policy Portal")
    
    st.markdown("""
    <div class='card'>
        <h3>About This Dashboard</h3>
        <p>
        This platform helps employees of <b>Avelora Consulting</b> easily access, understand, 
        and navigate company policies using AI-powered assistance.
        </p>

        <ul>
            <li>Browse official company policies</li>
            <li>Upload a policy temporarily and ask questions</li>
            <li>Use the AI assistant for HR and policy queries</li>
            <li>View your personal analytics and FAQs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------- ALL POLICIES ----------
elif page == "All Policies":
    st.title("📚 All Company Policies")
    files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".pdf")]

    if not files:
        st.info("No policies found. Add PDFs into the 'policies/' folder.")
    else:
        selected = st.selectbox("Select a policy:", files)
        show_policy_card(os.path.join(POLICY_DIR, selected))

# ---------- UPLOAD OR CHOOSE ----------
elif page == "Upload or Choose & Ask":
    st.title("📤 Upload or Choose a Policy")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload Policy (Temporary)", type=["pdf"])
    with col2:
        files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".pdf")]
        selected = st.selectbox("Choose Existing Policy", files if files else ["None"])

    chosen_file = uploaded if uploaded else (selected if selected != "None" else None)
    content = None

    if uploaded:
        try:
            reader = PdfReader(uploaded, strict=False)
            content = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except:
            st.error("Could not read uploaded file.")
    elif selected != "None":
        with open(os.path.join(POLICY_DIR, selected), "rb") as f:
            reader = PdfReader(f, strict=False)
            content = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())

    if content:
        st.markdown("## 💬 Ask Questions About This Policy")
        user_q = st.text_input("Your question:")
        c1, c2 = st.columns(2)
        with c1: ask_btn = st.button("Ask AI")
        with c2: sum_btn = st.button("Summarize Policy")

        if ask_btn:
            with st.spinner("Analyzing..."):
                ans = ask_ai(f"Policy:\n{content[:6000]}\n\nEmployee Question: {user_q}")
                st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {user_q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble-bot'><b>AI:</b> {ans}</div>", unsafe_allow_html=True)
                save_query(uploaded.name if uploaded else selected, user_q, ans)

        if sum_btn:
            with st.spinner("Summarizing..."):
                sm = ask_ai(f"Summarize this policy in 5 bullet points:\n{content[:6000]}")
                st.markdown(f"<div class='chat-bubble-bot'><b>Summary:</b><br>{sm}</div>", unsafe_allow_html=True)

# ---------- ASK POLICY AI ----------
elif page == "Ask Policy AI":
    st.title("💬 General Policy Assistant")
    q = st.text_area("Ask your HR or general policy question:")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            ans = ask_ai(q)
            st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-bot'><b>AI:</b> {ans}</div>", unsafe_allow_html=True)
            save_query("General", q, ans)

# ---------- ANALYTICS ----------
elif page == "My Analytics":
    st.title("📊 Your Analytics")
    df = pd.read_csv(QUERY_FILE)
    if df.empty:
        st.info("No questions asked yet.")
    else:
        st.metric("Total Questions", len(df))
        st.metric("Unique Policies Accessed", df['context'].nunique())

        wc_text = " ".join(df["question"])
        wc = WordCloud(width=600, height=300, background_color="white", colormap="Purples").generate(wc_text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        st.dataframe(df.sort_values("timestamp", ascending=False))

# ---------- FAQ ----------
elif page == "My FAQs":
    st.title("❓ Frequently Asked Questions")
    df = pd.read_csv(QUERY_FILE)
    if df.empty:
        st.info("Ask some questions first!")
    else:
        combined_questions = "\n".join(df["question"])
        with st.spinner("Generating FAQs..."):
            faqs = ask_ai(f"Generate 5 FAQ Q&A from this list:\n{combined_questions}")
        st.markdown(f"<div class='card'>{faqs}</div>", unsafe_allow_html=True)