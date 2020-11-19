#Szükséges csomagok importálása
import tensorflow as tf
import cv2
import numpy as np
import os
import pandas as pd
import random
from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *

#Mapping és modell betöltése
mapping = pd.read_csv("emnist-balanced-mapping.txt", delimiter = ' ', index_col=0, header=None, squeeze=True)
model=tf.keras.models.load_model('char_reader.model')

#Visszatérő változók definiálása
width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0,128,0)
true = 0
false = 0

#Tesztelés függvény
def testing():
    #Kép beolvasása és transzformálása
    img = cv2.imread('tmp.png',0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img,(28,28))
    img = img.reshape(1,28,28,1)
    img = img.astype('float32')
    img = img/255.0
    prediction=model.predict(img)
    return prediction

#Rajzolás a képernyőre
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

#Rajz alapján karakter tippelése
def tipp():
    filename = "tmp.png"
    image1.save(filename)
    prediction=testing()
    txt.insert(tk.INSERT,"{}\nValószínűség: {}%".format(chr(mapping[np.argmax(prediction[0])]),round(prediction[0][np.argmax(prediction[0])]*100,3)))

#Képernyő törlése
def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)

#Main program indítása
program = Tk()
program.resizable(0,0)
cv = Canvas(program, width=width, height=height, bg="red")
cv.pack()

# Pillow készít egy képet a háttérben, de ez csak a memóriában található meg
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

#Alsó szövegdoboz
txt=tk.Text(program,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
            padx=10,pady=10,height=5,width=20)
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

#Gombok rögzítése és program indítása
tippButton=Button(text="Tippelés",command=tipp)
clearButton=Button(text="Törlés",command=clear)
tippButton.pack()
clearButton.pack()
txt.pack()

program.title('Írott karakter felismerő')
program.mainloop()
# %%
