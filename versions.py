import numpy as np
import pandas as pd
import streamlit as st
import cv2
import tensorflow as tf
from importlib.metadata import version

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Streamlit:", st.__version__)
print("OpenCV:", cv2.__version__)
print("TensorFlow:", tf.__version__)
print("Drawable Canvas:", version("streamlit-drawable-canvas"))