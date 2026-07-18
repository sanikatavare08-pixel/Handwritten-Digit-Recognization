import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# ---------------- Load Model ----------------
import os
import gdown
from tensorflow.keras.models import load_model

FILE_ID = "1fSLHyKh8-l3_9AM_-61V90QNLAkncqgm"

if not os.path.exists("mask_final.keras"):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        "mask_final.keras",
        quiet=False
    )

model = load_model("mask_final.keras")


#model = load_model("https://drive.google.com/file/d/1fSLHyKh8-l3_9AM_-61V90QNLAkncqgm/view?usp=drive_link")

# ---------------- Session State ----------------
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#EAF6FF,#FFFFFF);
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#0F52BA;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:25px;
}

#.card{
  #  background:white;
   # padding:20px;
   # border-radius:20px;
   # box-shadow:0px 5px 18px rgba(0,0,0,0.15);
}

.result-good{
    background:#D4EDDA;
    color:#155724;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.result-bad{
    background:#F8D7DA;
    color:#721C24;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

div.stButton > button{
    width:100%;
    height:50px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#0F52BA;
    color:white;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown(
    "<div class='title'>😷 Face Mask Detection using CNN</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload an image or use your camera to detect whether a face is wearing a mask.</div>",
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
with st.sidebar:

    st.title("📌 About Project")

    st.info("""
This application uses a **Convolutional Neural Network (CNN)** to classify whether a person is wearing a face mask.

### Features
- 📂 Upload Image
- 📸 Live Camera
- ⚡ Fast Prediction
- 📊 Confidence Score

Developed using **TensorFlow** and **Streamlit**.
""")

# ---------------- Prediction Function ----------------
def predict_mask(img):

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    probability = prediction[0][0]

    return probability

# ---------------- Layout ----------------
left, right = st.columns([1.3,1])

# ================= LEFT =================
with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload Face Image",
        type=["jpg", "jpeg", "png"]
    )

    # ---------- Uploaded Image Prediction ----------
    if uploaded_file is not None:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        prob = predict_mask(img)

        st.write("### Prediction")

        if prob > 0.5:

            st.markdown(
                "<div class='result-bad'>❌ WITHOUT MASK</div>",
                unsafe_allow_html=True
            )

            confidence = prob

        else:

            st.markdown(
                "<div class='result-good'>✅ WITH MASK</div>",
                unsafe_allow_html=True
            )

            confidence = 1 - prob

        st.write("### Confidence")

        st.progress(float(confidence))

        st.metric("Confidence", f"{confidence:.2%}")

    # ---------- Camera Buttons ----------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Open Camera"):
            st.session_state.open_camera = True

    with col2:
        if st.button("❌ Close Camera"):
            st.session_state.open_camera = False

# ---------------- Camera ----------------
if st.session_state.open_camera:

    st.markdown("---")

    camera_image = st.camera_input("📷 Capture Image")

    if camera_image is not None:

        img = Image.open(camera_image)

        st.image(
            img,
            caption="Captured Image",
            use_container_width=True
        )

        prob = predict_mask(img)

        st.write("### Prediction")

        if prob > 0.5:

            st.markdown(
                "<div class='result-bad'>❌😊WITHOUT MASK</div>",
                unsafe_allow_html=True
            )

            confidence = prob

        else:

            st.markdown(
                "<div class='result-good'>✅😷WITH MASK</div>",
                unsafe_allow_html=True
            )

            confidence = 1 - prob

        st.write("### Confidence")

        st.progress(float(confidence))

        st.metric("Confidence", f"{confidence:.2%}")

# ---------------- Footer ----------------
st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;'>
Made by ❤️  <b>Sanika</b> <b>Tavare</b>
</div>
""",
unsafe_allow_html=True
)