This repository models the text of ["El ingenioso hidalgo don Quijote de la Mancha"](https://es.wikipedia.org/wiki/Don_Quijote_de_la_Mancha) using a Recurrent Neural Network. The final goal is to generate some new sentences.

Further details on the modeling and the solution are found in (this blog post)[www.jsaezgallego.com].


# Example of the generated sequence

Generate random characters with possibly non-existing words:
```
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5
```


An example of the output:

> un punto que el lo pueda en el campo la paraba, mi las armas, y que andando tu amo de ser dellos de tu rodela, que, decia que le la sido en el mundo. no le hallara en tus armas, y asi lo que decia: di senor del puerto 


Generate random characters and make sure the words exist in the book:

```python
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5 -o True
```

>  a aquel sobre el los dias de la mancha, don quijote de la mancha  estando en el campo; y a lo que era parte de la caballeria, la habeis de ser que le dejase de san benito que le habia dicho, de monte te puede caballero don quijote 


# Train the model

The model has been trained on a 61 GB GPU for roughly 5 hours. I used [Floydhub](https://www.floydhub.com/) to run it on the cloud. I made a mini-tutorial in the (blog post)[www.jsaezgallego.com]


# Ideas on how to improve the model

- Train for longer epochs with a bigger training set. This is a safe way of getting better results
- Implement a beam search algorithm character-wise
- Check if the generated word exists on a bigger dictionary
- Model whole words instead of characters
- Generate full sentences with a bi-directional RNN. More reasonable sentences will be generated for sure.
- Apply a pre-trained embedding matrix, even though with an old-style of writting as Don Quijote it will most likely not work very well
