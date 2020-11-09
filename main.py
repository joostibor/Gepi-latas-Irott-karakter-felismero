#Szükséges csomagok importálása
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

#Dataset betöltése és trainingje
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Adatok normalizálása
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

#Modell készítése, rétegek hozzáadása
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) #bementi réteg
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax')) #kimeneti réteg

#Modell lefordítása, tanítása és mentése
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

#Képek beolvasása a teszthez
for x in range(0, 20):
    readimg = cv2.imread(f'TestNumbers/{x}.png')[:,:,0]
    readimg = np.invert(np.array([readimg]))
    prediction = model.predict(readimg)
    print(f'The prediction of the sotfware is: {np.argmax(prediction)}')