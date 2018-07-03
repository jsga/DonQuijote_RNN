
# TODO: argparse

# load the network weights
filename = "weights-improvement-2L-20-1.5681.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, temperature=1.0):
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
	index = sample(prediction[0],0.34)
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

