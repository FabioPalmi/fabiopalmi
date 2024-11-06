"""
reteneurale.py

Contiene l'implementazione della SGD(stochastic gradient descent),
algoritmo di apprendimento per una rete neurale di tipo feedforward.
I gradienti verranno calcolati tramite Retropropagazione (Backpropagation)
"""

### Librerie
# Standard
import random
import time

# Terze-Parti
import numpy as np

class ReteNeurale(object):

    def __init__(self, sizes):
        """``sizes`` è una lista che contiene il numero di neuroni all'interno
        di ogni layer. Quindi [784, 16, 12, 10] sta a indicare che la rete è compostaif the list
        da 4 layer, con il primo(quello degli input) contenente 784 neuroni,
        il secondo 16 neuroni, il terzo 12 neuroni, e il quarto(quello di output) 10 neuroni.
        I bias e i pesi della rete sono inizializzati in modo randomico usando una
        distribuzione Gaussiana con media 0 e varianza 1.
        NB: Al primo layer, essendo di input, non verrà assegnato nessun valore
        bias o peso, in quanto questi vengono utilizzati solo per calcolare gli output
        dei layer successivi."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1)
                        for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Restituisce l'output della rete, con ''a'' il nostro input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, lr,
            test_data):
        """Addestra la rete neurale usando un algoritmo Gradient Descent Stocastico basato su mini-batch.stochastic
        Il ``training_data`` è una lista di tuple ''(x, y)'' che rappresentano gli input di addestramento gli
        output desiderati.
        Il ''test_data'' è invece utilizzata per testare la nostra rete neurale.
        Gli altri parametri sono:
        -''epochs'', numero di passaggi completi sul dataset.
        -''mini_batch_size'', grandezza del campione di esempi prima di un aggiornamento.
        -''lr'', learning rate."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            time2 = time.time()
            if test_data:
                print("Epoca {0}: {1} / {2}, tempo impiegato: {3:.2f} secondi".format(
                    j, self.evaluate(test_data), n_test, time2-time1))
            else:
                print("Epoca {0} completata in {1:.2f} secondi".format(j, time2-time1))

    def update_mini_batch(self, mini_batch, lr):
        """Aggiorna i bias e i pesi della rete applicando il Gradient Descent
        usando l'algoritmo di backpropagation ad un singolo mini-batch.
        Il ``mini_batch`` è una lista di tuple ``(x, y)``, e ``lr``
        è la learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(lr/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Restituisce una tupla ``(nabla_b, nabla_w)`` che rappresentano il
        gradiente della funzione costo C_x.``nabla_b`` e ``nabla_w``
        sono liste layer-a-layer di array numpy, simili a
        ''self.biases'' e ''self.weights''."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # lista che conserva tutte le attivazioni per ogni layer
        zs = [] # lista che conserva tutti i vettori z per ogni layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 è l'ultimo layer di neuroni l = 2 is the
        # l = 2 è il penultimo etc...
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Restituisce il numero di input di test dei quali la rete neurale dà
        risultato corretto in output. L'output di una rete neurale non è altro che
        l'indice del neurone che nel layer di output(quello finale) ha la più alta activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Restituisce il vettore delle derivate parziali"""
        return (output_activations-y)

#### Funzioni Delle Sigmoidi
def sigmoid(z):
    """Funzione Sigmoide."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivata della Funzione Sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))