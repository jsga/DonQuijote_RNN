This repository models the text of ["El ingenioso hidalgo don Quijote de la Mancha"](https://es.wikipedia.org/wiki/Don_Quijote_de_la_Mancha) using a Recurrent Neural Network. The final goal is to generate some new sentences.

Work in progress!

# Example
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5

# Ideas on how to improve the model

- Check that the generated word exists. Otherwise, reject and re-create
- Beam search character-wise
