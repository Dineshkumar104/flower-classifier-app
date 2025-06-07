# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # Title
# st.title("Image Classification using Custom CNN Model")

# # Load your trained model
# @st.cache_resource
# def load_cnn_model():
#     model = load_model("model.h5")
#     return model

# model = load_cnn_model()

# # Class labels (customize this according to your model)
# class_names = ['daisy', 'rose', 'sunflower','dandelion']  # Replace with your actual class names

# # Upload image
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Show the uploaded image
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image (modify input size if your model expects a different one)
#     # Preprocess image
#     img = img.resize((200, 200))  # Must match the model's input shape
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     # Prediction
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = np.max(prediction) * 100

#     # Output
#     st.subheader("Prediction:")
#     st.write(f"**Class:** {predicted_class}")
#     st.write(f"**Confidence:** {confidence:.2f}%")


import streamlit as st
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- DATABASE SETUP ---
conn = sqlite3.connect('users.db')
c = conn.cursor()

def create_users_table():
    c.execute('CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY, password TEXT)')
    conn.commit()

def add_user(email, password):
    c.execute('INSERT INTO users(email, password) VALUES (?, ?)', (email, password))
    conn.commit()

def login_user(email, password):
    c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
    return c.fetchone()

# --- LOAD MODEL ---
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()
class_names = ['daisy', 'rose', 'sunflower','dandelion']

# --- SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- MAIN PAGE ---
def main():
    st.title("🌼 Flower Classifier Login App")
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    create_users_table()

    if choice == "Register":
        st.subheader("Create New Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            add_user(email, password)
            st.success("You have successfully registered! You can now login.")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.success(f"Welcome, {email}!")
            else:
                st.error("Invalid login credentials")

    # --- IF LOGGED IN ---
    if st.session_state.logged_in:
        run_classifier()

# --- CLASSIFIER FUNCTION ---
def run_classifier():
    st.subheader("Upload Image for Flower Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.subheader("Prediction:")
        st.write(f"**Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

# --- RUN APP ---
if __name__ == '__main__':
    main()
