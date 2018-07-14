import argparse

ap = argparse.ArgumentParser(description='Generate new sentences from El Quijote, in Spanish.')
ap.add_argument("-f", "--filename", required=False,default='data/Don_Quijote_1-10Chap.txt',
				help="path to text. Used for training and for initializing random sequences.")
ap.add_argument("-w", "--weights", required=False,
				help="path to weights of the model. If empty the model will be trained. (warning: long time to compute!)")
ap.add_argument("-l", "--length_gen", required=False,type=int, default=1000,
				help="length of the generated sentence")
ap.add_argument("-o", "--whole_words", required=False,type=bool, default=False,
				help="True if whole existing words must be generated. False if new words can be generated.")
args = vars(ap.parse_args())

# For testing only
args = {'filename': 'data/Don_Quijote_1-10Chap.txt', 'weights': 'weights-improvement-3L-512-23-1.2375.hdf5', 'length_gen': 1000, 'whole_words':True}


# import homemade utils
from utils_generate_char_rnn import load_book_preprocess, model_define, sample, generate_words, generate_words_whole

# Load book
dataX, dataY, n_patterns, chars, n_chars, n_vocab, raw_text = load_book_preprocess(args['filename'],seq_length = 100)

# If weights are not given then train model (warning: takes a long time!)
if args['weights'] is None:
	# Create model and train
	model = model_define(dataX, dataY, n_patterns, n_vocab, do_train=True)

else:
	# Create KERAS model without training
	model = model_define(dataX, dataY, n_patterns, n_vocab, do_train = False)
	# load weigths
	model.load_weights(args['weights'])

# Generate characters or full words
if args['whole_words'] is False:
	generate_words(model,chars,n_vocab, dataX,seq_length=100)
else: # full existing words
	generate_words_whole(model,chars,n_vocab, dataX,raw_text,seq_length=100)
