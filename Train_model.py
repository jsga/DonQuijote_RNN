# Inspired from here https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import unidecode


# Load data
# The orignal text was downloaded from http://www.gutenberg.org/cache/epub/2000/pg2000.txt
# Headline and concluding paragraphs manually deleted.
# TODO: Download and process inside a function

# load ascii text and covert to lowercase
# filename = "Don_Quijote_Full.txt"
filename = "Don_Quijote_1-10Chap.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# Preprocess accents (very common in Spanish) to non-accented characters.
raw_text = unidecode.unidecode(raw_text)

# Delete possible repetitions of \n\n\n # TODO: code a small function to do this
raw_text = raw_text.replace('\n\n','\n')
raw_text = raw_text.replace('\n\n\n','\n')
raw_text = raw_text.replace('\n\n\n\n','\n')
raw_text = raw_text.replace('\n\n\n\n\n','\n')
raw_text = raw_text.replace('\n\n\n\n\n\n','\n')


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
print(char_to_int)


# Summary
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)




# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(256))
model.add(Dropout(0.25))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# ModelCheckpoint: Save weights at iterations
filepath="weights-improvement-2L-512-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit the model
history_model = model.fit(X, y, epochs=100, batch_size=64, callbacks=callbacks_list)


# n enano hacia senal de su venida; y asi, con
# estrano contento, llego a la venta y a las damas, las cualdo las armas del quieo las armas del mas armas y dasta de la mancha, ee cual aliuna merced, estaba con tu hasta que se labia de suien de caballeria que hecha alguna de su
# ropten de la mancha, el cual se la valta armado con que le debia su aabrada ee caballeria que ee con quijote de la mancha, el cual asuello y ceseacio de la puerta de lo que habia de su gabiendo. y que el aquella manza de su aaballeriza, sin que aldun yelta aqui alguna los di caballeros andantes senoras de lanza y dln la del cona tanta armado con ll que debia del cnmo alguna deseradion que so lo que deseaba, que el sodo el mundo daballero andantes las armas del mundo con la manza de la renta anui estaba, y que el sarte que se le daballeriza don quijote de la mancha, ee cual sando en la mancha, se oo debia de su caballo, y el mas ce la cual de caballeria que sen corenaia, casat de la manza que el cueara y asi, sen que el cuel se las armado caballero, pue era un mos daballeros andantes la pindad que duta a la dela de las armas duando se lo pue de la vanta a los que le daballero de auentura dl clleron contento y aquil de la venta anuella noche con la caballeriza de la pena y auena lo habia de sus armas que la vanta armado con qui alguna comtanta y don que habia hecho en la caballeriza dnn quijote de la fabeza en las armas de su caballo, y que el su renia ma desa de la venta, por el cien sobre aruella nanzi de la mancha, el cual estaba a los aaballeros andantes, y asi, le dejo de la mancha, el cual se la lanaha de su arraria y a sia manza, y, aui dsto a camioar, y a so mas de castello ee haberle conta la puejera de que de su podina y a
# las desadas en el mundo. y asi, con todo el mas cesasias que la cabaza ee cueea la pena de su reca y en la caballeria y donde la manza de la caballeria que en no puiio con quijote le manaia, som el para que eec sue en ma manza de la tenta a la vinta de la mancha, y que en todo el mas euerendiendo y aammo de la falta de su caballo, y que aruello la visa de su caballeriz
