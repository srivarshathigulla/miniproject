import streamlit as st
import pymongo
from datetime import datetime
from huggingface_hub import InferenceClient

# ----- CONFIGURATION -----
st.set_page_config(page_title="Hugging Face Chatbot App", layout="wide")

# Hugging Face Inference Client


hf_client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="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"
)
# MongoDB Setup
MONGODB_URI = st.secrets.get("MONGODB_URI")
mongo_client = pymongo.MongoClient(MONGODB_URI)
db = mongo_client["chatbot_app"]
chat_collection = db["chat_history"]
contact_collection = db["contact_messages"]

# Helper: Format prompt for Zephyr
def format_zephyr_prompt(messages):
    prompt = ""
    for msg in messages:
        role_tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
        prompt += f"{role_tag}\n{msg['content']}\n"
    prompt += "<|assistant|>\n"
    return prompt

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Chatbot", "About", "Contact"])

# ----- HOME -----
if page == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://jelvix.com/wp-content/uploads/2024/10/Banner-1800-3.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Welcome to Chatbot 🧠")
    st.markdown("Navigate using the sidebar to explore the Chatbot, About, or Contact pages.")

# ----- CHATBOT -----
elif page == "Chatbot":
    st.title("🤖 Chatbot")

    if "show_chat" not in st.session_state:
        st.session_state.show_chat = False

    if st.button("💬 Open Chatbot"):
        st.session_state.show_chat = not st.session_state.show_chat

    if st.session_state.show_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        prompt = st.chat_input("Say something...")

        if prompt:
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            full_prompt = format_zephyr_prompt(st.session_state.messages)

            with st.spinner("Thinking..."):
                response = hf_client.text_generation(
                    prompt=full_prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    stop_sequences=["<|user|>"]
                )
            reply = response.strip()

            st.chat_message("assistant").write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

            chat_collection.insert_one({
                "timestamp": datetime.utcnow(),
                "session": st.session_state.get("session_id", "default"),
                "messages": st.session_state.messages.copy()
            })

# ----- ABOUT -----
elif page == "About":
    st.title("About the Chatbot 🤓")
    st.markdown("""
    Welcome to our chatbot! 
    This chatbot uses `Zephyr-7B-Beta` via Hugging Face Inference API for interactive AI conversations.
    It's lightweight enough to run on a t2.micro instance with MongoDB integration.
    We're here to make your experience smoother, faster, and more engaging.
    Whether you’re looking for answers, support, or just exploring, our intelligent assistant is designed to help you 24/7.
    """)
    st.video("https://youtu.be/FCcg1L0PCD4")

# ----- CONTACT -----
elif page == "Contact":
    st.title("📬 Contact Us")

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send")

        if submitted:
            contact_collection.insert_one({
                "name": name,
                "email": email,
                "message": message,
                "timestamp": datetime.utcnow()
            })
            st.success("Thank you! Your message has been sent.")

    st.markdown("### Our Location 📍")
    st.components.v1.iframe(
        src="https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d15157.62977270674!2d79.3579143344171!3d18.237134304838456!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e0!3m2!1sen!2sin!4v1744707889779!5m2!1sen!2si",
        height=400,
        width=700,
    )
