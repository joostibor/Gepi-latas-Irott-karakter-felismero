#Szükséges csomagok importálása
import tensorflow as tf
import cv2
import numpy as np
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

#Távolról, lámpafénynél fotózott fehér hátterű teszt
files = os.listdir('TestChars/FarLight')
print('---------------Távoli, lámpafényes teszt---------------')

for f in files:
    readimg = cv2.imread(f'TestChars/FarLight/{f}')[:,:,0]
    readimg = cv2.bitwise_not(readimg)
    readimg = cv2.resize(readimg, (28,28))
    ret,readimg = cv2.threshold(readimg,127,255,cv2.THRESH_BINARY)
    readimg = readimg.reshape(-1, 28, 28, 1)
    prediction = char_model.predict(readimg)
    print(f'Char: {f[0]}, The prediction: {chr(mapping[np.argmax(prediction[0])])}')
    if f[0] == chr(mapping[np.argmax(prediction[0])]):
        ok+=1
    else:
        notok+=1

print('Jó tippek: ', ok)
print('Rossz tippek: ', notok)

#Vakuval fotózott fehér hátterű teszt
files = os.listdir('TestChars/LightWhite')
print('---------------Vakuzott, fehér hátterű teszt---------------')
ok = 0
notok = 0

for f in files:
    readimg = cv2.imread(f'TestChars/LightWhite/{f}')[:,:,0]
    readimg = cv2.bitwise_not(readimg)
    readimg = cv2.resize(readimg, (28,28))
    ret,readimg = cv2.threshold(readimg,127,255,cv2.THRESH_BINARY)
    readimg = readimg.reshape(-1, 28, 28, 1)
    prediction = char_model.predict(readimg)
    print(f'Char: {f[0]}, The prediction: {chr(mapping[np.argmax(prediction[0])])}')
    if f[0] == chr(mapping[np.argmax(prediction[0])]):
        ok+=1
    else:
        notok+=1

print('Jó tippek: ', ok)
print('Rossz tippek: ', notok)

#Távolról fotózott színes hátterű teszt
files = os.listdir('TestChars/FarLightColor')
print('---------------Távolról fotózott színes hátterű teszt---------------')
ok = 0
notok = 0

for f in files:
    readimg = cv2.imread(f'TestChars/FarLightColor/{f}')[:,:,0]
    readimg = cv2.bitwise_not(readimg)
    readimg = cv2.resize(readimg, (28,28))
    ret,readimg = cv2.threshold(readimg,127,255,cv2.THRESH_BINARY)
    readimg = readimg.reshape(-1, 28, 28, 1)
    prediction = char_model.predict(readimg)
    print(f'Char: {f[0]}, The prediction: {chr(mapping[np.argmax(prediction[0])])}')
    if f[0] == chr(mapping[np.argmax(prediction[0])]):
        ok+=1
    else:
        notok+=1

print('Jó tippek: ', ok)
print('Rossz tippek: ', notok)

#Vakuval fotózott színes hátterű teszt
files = os.listdir('TestChars/ColorBG')
print('---------------Vakuval fotózott színes hátterű teszt---------------')
ok = 0
notok = 0

for f in files:
    readimg = cv2.imread(f'TestChars/ColorBG/{f}')[:,:,0]
    readimg = cv2.bitwise_not(readimg)
    readimg = cv2.resize(readimg, (28,28))
    ret,readimg = cv2.threshold(readimg,127,255,cv2.THRESH_BINARY)
    readimg = readimg.reshape(-1, 28, 28, 1)
    prediction = char_model.predict(readimg)
    print(f'Char: {f[0]}, The prediction: {chr(mapping[np.argmax(prediction[0])])}')
    if f[0] == chr(mapping[np.argmax(prediction[0])]):
        ok+=1
    else:
        notok+=1

print('Jó tippek: ', ok)
print('Rossz tippek: ', notok)
