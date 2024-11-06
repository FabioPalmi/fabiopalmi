"""
svm_prova.py

Un classificatore SVM di prova per riconoscere le cifre dal dataset MINST.
"""
### Librerie
# Terze Parti
from sklearn import svm

class SvmProva(object):

    def __init__(self):
        "Classe per SVM di prova"

    def svm_di_prova(self, training_data, test_data):
        # addestramento
        print("Inizio Addestramento!")
        print("Attendere...")
        clf = svm.SVC()
        clf.fit(training_data[0], training_data[1])
        print("Addestramento Completato!")
        # test
        print("Inizio Fase di Test!")
        print("Attendere...")
        predictions = [int(a) for a in clf.predict(test_data[0])]
        num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
        print("Fase di Test Completata!")
        print("{0} of {1} valori corretti!.".format(num_correct, len(test_data[1])))

