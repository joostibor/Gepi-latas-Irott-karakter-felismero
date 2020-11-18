#Szükséges csomagok importálása
#%%
import tensorflow as tf
import extra_keras_datasets as extkds
import gzip
import numpy as np
import struct

#Dataset betöltése
def read_idx(filename):
    print('Processing data from %s.' % filename)
    with gzip.open(filename, 'rb') as f:
        z, dtype, dim = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_emnist():
    train_images = 'train-images-idx3-ubyte.gz'
    train_labels = 'train-labels-idx1-ubyte.gz'
    test_images = 't10k-images-idx3-ubyte.gz'
    test_labels = 't10k-labels-idx1-ubyte.gz'

    train_X = read_idx(train_images)
    train_y = read_idx(train_labels)
    test_X = read_idx(test_images)
    test_y = read_idx(test_labels)
    return (train_X, train_y, test_X, test_y)

x_train, y_train, x_test, y_test = load_emnist()

#Adatok normalizálása
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Modell készítése, rétegek hozzáadása
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #bementi réteg
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #kimeneti réteg

#Modell lefordítása, tanítása és mentése
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#Modell mentése
model.save('char_reader.model')
# %%
