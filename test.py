#%%
#Szükséges csomagok importálása
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#Modell betöltése
char_model = tf.keras.models.load_model('char_reader.model')

#Karakterfelismerő teszt
#Változók a statisztikához
ok = 0
notok = 0
alltest = 0
shadowtestwhite = 0

#Osztályozás betöltése
mapping = pd.read_csv("emnist-balanced-mapping.txt", delimiter = ' ', index_col=0, header=None, squeeze=True)

#Árnyékos fehér hátterű teszt
files = os.listdir('TestChars/ShadowWhite')
chars = [f[3] for f in files]


# %%
