import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import unidecode

# Inspired from here https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
def load_book_preprocess(filename,seq_length = 100):
	# Load data
	# The orignal text was downloaded from http://www.gutenberg.org/cache/epub/2000/pg2000.txt
	# Headline and concluding paragraphs manually deleted.

	# load ascii text and covert to lowercase
	# filename = "Don_Quijote_Full.txt"
	#filename = "Don_Quijote_1-10Chap.txt"

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

	# Print summary
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)

	# prepare the dataset of input to output pairs encoded as integers
	#seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print("Total Patterns: ", n_patterns)

	return dataX, dataY, n_patterns, chars, n_chars, n_vocab


def model_define(dataX, dataY, n_patterns, n_vocab,seq_length = 100, do_train = False):

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

	# Save model to YAML
	# model_yaml = model.to_yaml()
	# with open("model.yaml", "w") as yaml_file:
	# 	yaml_file.write(model_yaml)

	# ModelCheckpoint: Save weights at iterations
	filepath="/output/weights-improvement-2L-512-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# Fit the model
	if do_train is True:
		model.fit(X, y, epochs=100, batch_size=64, callbacks=callbacks_list)

	return model



def generate_words(model,chars,n_vocab, dataX,seq_length):

	int_to_char = dict((i, c) for i, c in enumerate(chars))

	# Define sampling strategy
	def sample(preds, temperature=0.4):
		# helper function to sample an index from a probability array
		preds = np.asarray(preds).astype('float64')
		preds = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)
		return np.argmax(probas)


	# pick a random seed
	start = np.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print("Seed:")
	print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


	# generate characters
	for i in range(2000):
		pattern_aux = pattern[(len(pattern) - seq_length):len(pattern)]
		x = np.reshape(pattern_aux, (1, len(pattern_aux), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0) # Predict probability of character appearing next
		#ndex = np.argmax(prediction)
		index = sample(prediction[0],0.2)
		#result = int_to_char[index]
		# add new element
		pattern.append(index)
		#pattern.pop(0)
		# delete first entry
		#pattern = pattern[1:len(pattern)]
	print("\nDone.")


	# Translate index to char
	seq_in = [int_to_char[value] for value in pattern]
	print(''.join(seq_in))

	return seq_in