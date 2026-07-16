#pip install opencv-python
#pip install streamlit-drawable-canvas


import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#141E30,#243B55);
}

.main-title{
text-align:center;
font-size:48px;
font-weight:800;
color:white;
}

.subtitle{
text-align:center;
color:#D1D5DB;
font-size:20px;
margin-bottom:30px;
}

.glass{
width:340px;
margin:auto;
padding:20px;
background:rgba(255,255,255,.12);
backdrop-filter:blur(20px);
border-radius:20px;
border:1px solid rgba(255,255,255,.2);
box-shadow:0 8px 25px rgba(0,0,0,.3);
}

.result-card{
width:400px;
margin:auto;
margin-top:25px;
padding:25px;
background:white;
border-radius:18px;
text-align:center;
box-shadow:0 10px 25px rgba(0,0,0,.25);
}

.big-digit{
font-size:80px;
font-weight:bold;
color:#4F46E5;
}

.stButton>button{
border-radius:50px;
font-size:18px;
font-weight:bold;
height:55px;
background:linear-gradient(90deg,#6366F1,#8B5CF6);
color:white;
border:none;
transition:.3s;
}

.stButton>button:hover{
transform:scale(1.05);
}
            
""", unsafe_allow_html=True)
# ---------------- Load Model ----------------
model = load_model("digit_recognition_model.keras")

# ---------------- Sidebar ----------------
st.sidebar.title("📌 Instructions")

st.sidebar.write("""
1. Draw a digit (0-9)

2. Click **Predict**

3. Model will recognize your digit

Made by ❤️ Sanika Tavare
""")

# ---------------- Title ----------------
st.markdown('<div class="main-title">✍️ Handwritten Digit Recognition</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Draw any digit from 0 to 9 and let AI predict it!</div>',
    unsafe_allow_html=True
)

# ---------------- Card ----------------
st.markdown('<div>', unsafe_allow_html=True)

canvas_result = st_canvas(
                        fill_color="#00000000",
                         stroke_width=10,
                         stroke_color="#FFFFFF",
                         background_color="#FFFFFFF",
                         width=280,
                         height=280,
                         drawing_mode="freedraw",
                         key='canvas'
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction ----------------
predict=st.button("🔍 Predict Digit")


if predict:

    if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, (28, 28))

        #gray = 255 - gray

        gray = gray / 255.0

        gray = gray.reshape(-1, 784)

        prediction = model.predict(gray)

        digit = np.argmax(prediction)


    confidence = float(np.max(prediction) * 100)

        

    st.markdown(f"""
    <div class="result-card">
        <h2>🎯 Prediction</h2>
        <div class="big-digit">{digit}</div>
        <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(confidence / 100)