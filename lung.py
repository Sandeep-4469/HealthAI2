# sudo apt update
# sudo apt install -y libgl1-mesa-glx libglib2.0-0
import tensorflow as tf
from tensorflow.keras.models import load_model 
import keras
import cv2 
import numpy as np
def lung(path):
  img = cv2.imread(path)
  d = {}
  # tb_model = load_model("TB.keras")
  # shape = tb_model.layers[0].input_shape[1:-1]
  # print(shape)
  # tb_img = cv2.resize(img,(28,28))
  # tb_img = np.expand_dims(tb_img, axis=0)
  # d["Tuberculosis"] = tb_model.predict(tb_img)

  pneumonia_model = load_model("PN.h5")
  shape = pneumonia_model.layers[0].input_shape[1:-1]
  pn_img = cv2.resize(img,(150,150))
  print(on_img.shape,"..................")
  pn_img = np.expand_dims(pn_img, axis=0)
  d["Pneumonia"] = pneumonia_model.predict(pn_img)

  covid_model = load_model("covid_sequential  (1).h5")
  shape = covid_model.layers[0].input_shape[1:-1]
  pn_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cd_img = cv2.resize(pn_img,shape)
  cd_img = np.expand_dims(cd_img, axis=0)
  d["Covid"] = covid_model.predict(cd_img)
  return d
lung("COVID-1.png")