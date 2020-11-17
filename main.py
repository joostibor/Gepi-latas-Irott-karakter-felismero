#Szükséges csomagok importálása
#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Dataset betöltése és trainingje
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Adatok normalizálása
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Modell készítése, rétegek hozzáadása
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #bementi réteg
model.add(tf.keras.layers.Dense(units=128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation= tf.nn.softmax)) #kimeneti réteg

#Modell lefordítása, tanítása és mentése
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs=5)
loss, accuracy = model.evaluate(x_test, y_test)

#Modell mentése
model.save('digits.mod')

#Képek beolvasása a teszthez
for x in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/{x}_big.png')[:,:,0]
    readimg = np.invert(np.array([readimg]))
    prediction = model.predict(readimg)
    print(f'Number: {x}, The prediction of the sotfware is: {np.argmax(prediction)}')
    plt.imshow(readimg[0], cmap=plt.cm.binary)
    plt.show()


for x in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/{x}.png')[:,:,0]
    readimg = np.invert(np.array([readimg]))
    prediction = model.predict(readimg)
    print(f'Number: {x}, The prediction of the sotfware is: {np.argmax(prediction)}')

for x in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/1{x}.png')[:,:,0]
    readimg = np.invert(np.array([readimg]))
    prediction = model.predict(readimg)
    print(f'Number: {x}, The prediction of the sotfware is: {np.argmax(prediction)}')
# %%
