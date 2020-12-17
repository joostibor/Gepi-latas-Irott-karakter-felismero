#Szükséges csomagok importálása
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

num_model = tf.keras.models.load_model('digits_reader.model')
char_model = tf.keras.models.load_model('char_reader.model')

#Karakterfelismerő teszt
#Változók a statisztikához
okc = 0
notokc = 0
allchar = 0
bigchar = 0
smallchar = 0

#Fájlbeolvasás karakter felismeréshez
mapping = pd.read_csv("emnist-balanced-mapping.txt", delimiter = ' ', index_col=0, header=None, squeeze=True)

allchar += 26
for c in range(97,123):
    readimg = cv2.imread(f'TestChars/{chr(c)}.png')[:,:,0]
    readimg = np.array([readimg])
    first_idx = readimg[0,0,0]
    for x in range(0,1):
        for y in range(0,28):
            for z in range(0,28):
                if readimg[x,y,z] == first_idx:
                    readimg[x,y,z] = 0
                else:
                    readimg[x,y,z] = 255
    readimg = readimg.reshape(-1, 28, 28, 1)
    prediction = char_model.predict(readimg)
    itemidx = 0
    for i in range(0, prediction.shape[1]): 
        if prediction[0, i] == 1:
            itemindex=i
    if chr(c) == chr(mapping[itemindex]):
        okc += 1
        smallchar += 1
    else:
        notokc += 1
    print(f'Char: {chr(c)}, The prediction of the sotfware is: {chr(mapping[itemindex])}')

allchar += 26
for c in range(65,91):
    readimg = cv2.imread(f'TestChars/{chr(c)}_b.png')[:,:,0]
    readimg = np.array([readimg])
    first_idx = readimg[0,0,0]
    for x in range(0,1):
        for y in range(0,28):
            for z in range(0,28):
                if readimg[x,y,z] == first_idx:
                    readimg[x,y,z] = 0
                else:
                    readimg[x,y,z] = 255
    readimg = readimg.reshape(-1, 28, 28, 1)
    prediction = char_model.predict(readimg)
    itemidx = 0
    for i in range(0, prediction.shape[1]): 
        if prediction[0, i] == 1:
            itemindex=i
    if chr(c) == chr(mapping[itemindex]):
        okc += 1
        bigchar += 1
    else:
        notokc += 1
    print(f'Char: {chr(c)}, The prediction of the sotfware is: {chr(mapping[itemindex])}')

#Számfelismerő teszt
#Változók a statisztikához
ok = 0
notok = 0
allnumber = 0
thicknums = 0
nastynums = 0
randnums = 0
finenums = 0

#Teszt
allnumber += 10
for img in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/{img}_big.png')[:,:,0]
    readimg = np.array([readimg])
    first_idx = readimg[0,0,0]
    for x in range(0,1):
        for y in range(0,28):
            for z in range(0,28):
                if readimg[x,y,z] == first_idx:
                    readimg[x,y,z] = 0
                else:
                    readimg[x,y,z] = 255
    prediction = num_model.predict(readimg)
    print(f'Number: {img}, The prediction of the sotfware is: {np.argmax(prediction)}')
    if img == np.argmax(prediction):
        ok += 1
        thicknums += 1
    else:
        notok += 1

allnumber += 10
for img in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/{img}.png')[:,:,0]
    readimg = np.array([readimg])
    first_idx = readimg[0,0,0]
    for x in range(0,1):
        for y in range(0,28):
            for z in range(0,28):
                if readimg[x,y,z] == first_idx:
                    readimg[x,y,z] = 0
                else:
                    readimg[x,y,z] = 255
    prediction = num_model.predict(readimg)
    print(f'Number: {img}, The prediction of the sotfware is: {np.argmax(prediction)}')
    if img == np.argmax(prediction):
        ok += 1
        nastynums += 1
    else:
        notok += 1

allnumber += 10
for img in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/1{img}.png')[:,:,0]
    readimg = np.array([readimg])
    first_idx = readimg[0,0,0]
    for x in range(0,1):
        for y in range(0,28):
            for z in range(0,28):
                if readimg[x,y,z] == first_idx:
                    readimg[x,y,z] = 0
                else:
                    readimg[x,y,z] = 255
    prediction = num_model.predict(readimg)
    print(f'Number: {img}, The prediction of the sotfware is: {np.argmax(prediction)}')
    if img == np.argmax(prediction):
        ok += 1
        randnums += 1
    else:
        notok += 1

allnumber += 10
for img in range(0, 10):
    readimg = cv2.imread(f'TestNumbers/{img}_c.png')[:,:,0]
    readimg = np.array([readimg])
    first_idx = readimg[0,0,0]
    for x in range(0,1):
        for y in range(0,28):
            for z in range(0,28):
                if readimg[x,y,z] == first_idx:
                    readimg[x,y,z] = 0
                else:
                    readimg[x,y,z] = 255
    prediction = num_model.predict(readimg)
    print(f'Number: {img}, The prediction of the sotfware is: {np.argmax(prediction)}')
    if img == np.argmax(prediction):
        ok += 1
        finenums += 1
    else:
        notok += 1

print('Karakter tesztelés szumma:')
print(f'Összes tipp: {allchar}')
print(f'Jó tippek: {okc}')
print(f'Rossz tippek: {notokc}')
print(f'Kisbetűk esetén: {smallchar}/26')
print(f'Nagybetűk esetén: {bigchar}/26')
print('Szám tesztelés szumma:')
print(f'Összes tipp: {allnumber}')
print(f'Jó tippek: {ok}')
print(f'Rossz tippek: {notok}')
print(f'Vastag írásmódú számok esetén: {thicknums}/10')
print(f'Ronda írású számok esetén: {nastynums}/10')
print(f'Odafigyelt írásmódú számok esetén: {randnums}/10')
print(f'Odafigyelt írásmódú és színes hátterű számok esetén: {finenums}/10')