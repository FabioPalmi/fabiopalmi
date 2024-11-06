"""
mnist_data.py

Libreria che serva a caricare i dati di MNIST.
"""

### Librerie
# Standard
import _pickle as pickle
import gzip

# Terze-Parti
import numpy as np

def load_data():
    """Restituisce i dati MNIST sottoforma di tuple."""
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Restituisce i dati MNIST sottoforma di tuple compatibili con
    l'implementazione utilizzata della rete neurale."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Restituisce un vettore a 10 dimensioni con 1.0 nella posizione j, e 0 in tutte le altre."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e