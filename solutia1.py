import numpy as np
import pandas as pd

class Nod:
    def __init__(self, atribut=None, prag=None, eticheta=None):
        self.atribut = atribut
        self.prag = prag
        self.eticheta = eticheta
        self.stanga = None
        self.dreapta = None

def entropie(y):
    valori, numarOcurente = np.unique(y, return_counts=True)
    probabilitati = numarOcurente / len(y)
    return -np.sum(probabilitati * np.log2(probabilitati))

def castigInformatie(y, yStanga, yDreapta):
    return entropie(y) - (len(yStanga) / len(y) * entropie(yStanga) + len(yDreapta) / len(y) * entropie(yDreapta))

def ceaMaiBunaImpartire(X, y):
    celMaiBunCastig = 0
    celMaiBunAtribut = None
    celMaiBunPrag = None

    for atribut in range(X.shape[1]):
        pragi = np.unique(X[:, atribut])
        for prag in pragi:
            mascaStanga = X[:, atribut] <= prag
            mascaDreapta = ~mascaStanga

            yStanga, yDreapta = y[mascaStanga], y[mascaDreapta]

            if len(yStanga) == 0 or len(yDreapta) == 0:
                continue

            castig = castigInformatie(y, yStanga, yDreapta)
            if castig > celMaiBunCastig:
                celMaiBunCastig = castig
                celMaiBunAtribut = atribut
                celMaiBunPrag = prag

    return celMaiBunAtribut, celMaiBunPrag

def construiesteArbore(X, y):
    if len(np.unique(y)) == 1:
        return Nod(eticheta=y[0])

    if X.shape[0] == 0:
        return None

    atribut, prag = ceaMaiBunaImpartire(X, y)

    if atribut is None:
        eticheta = np.bincount(y).argmax()
        return Nod(eticheta=eticheta)

    radacina = Nod(atribut=atribut, prag=prag)

    mascaStanga = X[:, atribut] <= prag
    mascaDreapta = ~mascaStanga

    radacina.stanga = construiesteArbore(X[mascaStanga], y[mascaStanga])
    radacina.dreapta = construiesteArbore(X[mascaDreapta], y[mascaDreapta])

    return radacina

def prezice(arbore, X):
    if arbore.eticheta is not None:
        return arbore.eticheta

    if X[arbore.atribut] <= arbore.prag:
        return prezice(arbore.stanga, X)
    else:
        return prezice(arbore.dreapta, X)

def codificaSold(sold):
    if 200 <= sold < 800:
        return 0
    elif 800 <= sold < 1400:
        return 1
    elif sold >= 1400:
        return 2
    else:
        return -1

caleFisierAntrenament = 'dateAntrenament.csv'
caleFisierValidare = 'dateValidare.csv'

dateAntrenament = pd.read_csv(caleFisierAntrenament)
print("Datele de antrenament au fost incarcate cu succes!")

XAntrenament = dateAntrenament.iloc[:, 1:-1].values
yAntrenament = dateAntrenament.iloc[:, -1].values

XAntrenament = XAntrenament.astype(float)
yAntrenament = np.array([codificaSold(s) for s in yAntrenament])

arbore = construiesteArbore(XAntrenament, yAntrenament)

dateValidare = pd.read_csv(caleFisierValidare)
print("Datele de validare au fost incarcate cu succes!")
print(dateValidare.head())

XValidare = dateValidare.iloc[:, 1:-1].values
yValidare = dateValidare.iloc[:, -1].values

XValidare = XValidare.astype(float)

corecte = 0
total = len(yValidare)

for exemplu, etichetaReal in zip(XValidare, yValidare):
    predicte = prezice(arbore, exemplu)
    if predicte == codificaSold(etichetaReal):
        corecte += 1

rataEroare = 1 - (corecte / total)

print(f"Rata de eroare: {rataEroare:.4f} ({corecte}/{total} corecte)")
