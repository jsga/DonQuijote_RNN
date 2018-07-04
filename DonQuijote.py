import argparse

ap = argparse.ArgumentParser(description='Generate new sentences from El Quijote, in Spanish.')
ap.add_argument("-f", "--filetext", required=False,
				help="path to text. Used for training and for initializing random sequences.")
ap.add_argument("-w", "--weights", required=False,default='',
				help="path to weights of the model")
ap.add_argument("-l", "--length", required=False,type=int, default=1000,
				help="length of the generated sentence")
args = vars(ap.parse_args())


# import utils
from utils_generate_char_rnn import load_book_preprocess, model_define, generate_words


# Load book
dataX, dataY, n_patterns, n_chars, n_vocab = load_book_preprocess(filename,seq_length = 100)

# If weights are not given then train model
if args['weights'] is None:
	# Create model and train
	model = model_define(dataX, dataY, n_patterns, n_vocab, do_train=True)

else:

	# Create KERAS model without training
	model = model_define(dataX, dataY, n_patterns, n_vocab, do_train = False)
	# load weigths
	model.load_weights(args['weights'])

# Generate characters
generate_words(model,chars, dataX,seq_length)
