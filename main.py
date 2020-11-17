#Szükséges csomagok importálása
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

num_model = tf.keras.models.load_model('digits_reader.model')

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
    plt.imshow(readimg[0], cmap=plt.cm.binary)
    plt.show()

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
    plt.imshow(readimg[0], cmap=plt.cm.binary)
    plt.show()

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
    plt.imshow(readimg[0], cmap=plt.cm.binary)
    plt.show()

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
    plt.imshow(readimg[0], cmap=plt.cm.binary)
    plt.show()

print('Összesen:')
print(f'Összes tipp: {allnumber}')
print(f'Jó tippek: {ok}')
print(f'Rossz tippek: {notok}')
print(f'Vastag írásmódú számok esetén: {thicknums}/10')
print(f'Ronda írású számok esetén: {nastynums}/10')
print(f'Hirtelen írásmódú számok esetén: {randnums}/10')
print(f'Odafigyelt írásmódú és színes hátterű számok esetén: {finenums}/10')
# %%
