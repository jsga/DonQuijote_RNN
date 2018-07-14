This repository models the text of ["El ingenioso hidalgo don Quijote de la Mancha"](https://es.wikipedia.org/wiki/Don_Quijote_de_la_Mancha) using a Recurrent Neural Network. The final goal is to generate some new sentences.

Further details on the modeling and the solution are found in (this blog post)[www.jsaezgallego.com].


# Example of the generated sequence

Generate random characters with possibly non-existing words:
```
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5
```


An example of the output:

> con todo eso, se le dejaron de ser su romance v me dejase, porque no le dejare y facilidad de su modo que de la lanza en la caballeriza, por el mesmo camino, y la donde se le habia de haber de los que el campo, porque el estaba la cabeza que le parece a le puerto y de contento, son de la primera entre algunas cosas de la venta, con tanta furia de su primer algunos que a los caballeros andantes a su lanza, y, aunque el no puede le dios te parecian y a tu parte, se dios ser puede los viera en la caballeria en la caballeria en altas partes de la mancha, 


Generate random characters and make sure the words exist in the book:

```python
python3 DonQuijote.py -w weights-improvement-3L-512-23-1.2375.hdf5 -o True
```

>  al cual le parecieron don quijote de la mancha, en cuando le daba a le senor tio en el corral, y tio que andaba muy acerto los dos viejos, y, al caso de van manera con el de tu escudero. don quijote y mas venta a su asno, con toda su amo pasa dios de la caballeria y de al que habia leido, no habia de ser tu escudero: la suelo del camino de la venta, de que san caballo de los que le habia dejado; a este libro es este es el mismo coche, como te ve don mucho deseos de los que el caballero le hallaba; y al corral con la cabeza que aquel sabio en la gente de la lanza y tan las demas y camas de tu escudero, 



# Train the model

The model has been trained on a 61 GB GPU for roughly 5 hours. I used [Floydhub](https://www.floydhub.com/) to run it on the cloud. I made a mini-tutorial in the (blog post)[www.jsaezgallego.com]


# Ideas on how to improve the model

- Train for longer epochs with a bigger training set. This is a safe way of getting better results
- Implement a beam search algorithm character-wise
- Check if the generated word exists on a bigger dictionary
- Model whole words instead of characters
- Generate full sentences with a bi-directional RNN. More reasonable sentences will be generated for sure.
- Apply a pre-trained embedding matrix, even though with an old-style of writting as Don Quijote it will most likely not work very well
