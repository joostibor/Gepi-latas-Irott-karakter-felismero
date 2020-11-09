#Szükséges csomagok importálása
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Dataset betöltése és trainingje
mnistnums = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnistnums.load_data()

#Adatok normalizálása
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Modell készítése, rétegek hozzáadása
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #bementi réteg
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #kimeneti réteg

#Modell lefordítása, tanítása és mentése
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.save('nums.mod')