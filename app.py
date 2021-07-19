import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Rotation
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))
operation = st.selectbox("Operation",(-30,-60,-90,-120))
import cv2 as cv
def import_and_predict(image,operation):
  image2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  # get the image shape
  rows, cols, dim = image2.shape
  if operation == -30:
    # transformation matrix for Rotation
    # translations applied to down
    angle = np.radians(operation)
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
    rotated_img = cv.warpPerspective(image2, M, (int(cols),int(rows)))
    st.image(rotated_img, use_column_width=True)
    return 0

  if operation == -60:
    # transformation matrix for Rotation
    # translations applied to down
    angle = np.radians(operation)
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
    rotated_img = cv.warpPerspective(image2, M, (int(cols),int(rows)))
    st.image(rotated_img, use_column_width=True)
    return 0

  if operation == -90:
    # transformation matrix for Rotation
    # translations applied to down
    angle = np.radians(operation)
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
    rotated_img = cv.warpPerspective(image2, M, (int(cols),int(rows)))
    st.image(rotated_img, use_column_width=True)
    return 0

  if operation == -120:
    # transformation matrix for Rotation
    # translations applied to down
    angle = np.radians(operation)
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
    rotated_img = cv.warpPerspective(image2, M, (int(cols),int(rows)))
    st.image(rotated_img, use_column_width=True)
    return 0
if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("See Rotation"):
  result=import_and_predict(image,operation)
  
if st.button("About"):
  st.header("RAJENDRA YADAV")
  st.subheader("Student of PIET")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)