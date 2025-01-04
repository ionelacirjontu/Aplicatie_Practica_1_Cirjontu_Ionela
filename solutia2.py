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

def codificaSubinterval0(sold):
    if 200 <= sold < 400:
        return 3
    elif 400 <= sold < 600:
        return 4
    elif 600 <= sold < 800:
        return 5
    else:
        return -1

def codificaSubinterval1(sold):
    if 800 <= sold < 1000:
        return 6
    elif 1000 <= sold < 1200:
        return 7
    elif 1200 <= sold < 1400:
        return 8
    else:
        return -1

def codificaSubinterval2(sold):
    if 1400 <= sold < 1800:
        return 9
    elif 1800 <= sold < 2200:
        return 10
    elif sold >= 2200:
        return 11
    else:
        return -1

caleFisierAntrenament = 'dateAntrenament.csv'
caleFisierValidare = 'dateValidare.csv'

dateAntrenament = pd.read_csv(caleFisierAntrenament)
XAntrenament = dateAntrenament.iloc[:, 1:-1].values.astype(float)
yAntrenament = np.array([codificaSold(s) for s in dateAntrenament.iloc[:, -1].values])

arborePrincipal = construiesteArbore(XAntrenament, yAntrenament)

XSecundar0 = XAntrenament[yAntrenament == 0]
ySecundar0 = np.array([codificaSubinterval0(s) for s in dateAntrenament.iloc[yAntrenament == 0, -1]])
arboreSecundar0 = construiesteArbore(XSecundar0, ySecundar0)

XSecundar1 = XAntrenament[yAntrenament == 1]
ySecundar1 = np.array([codificaSubinterval1(s) for s in dateAntrenament.iloc[yAntrenament == 1, -1]])
arboreSecundar1 = construiesteArbore(XSecundar1, ySecundar1)

XSecundar2 = XAntrenament[yAntrenament == 2]
ySecundar2 = np.array([codificaSubinterval2(s) for s in dateAntrenament.iloc[yAntrenament == 2, -1]])
arboreSecundar2 = construiesteArbore(XSecundar2, ySecundar2)

arboriSecundari = [arboreSecundar0, arboreSecundar1, arboreSecundar2]

def preziceFinal(arborePrincipal, arboriSecundari, exemplu):
    etichetaPrincipala = prezice(arborePrincipal, exemplu)
    if etichetaPrincipala in [0, 1, 2]:
        return prezice(arboriSecundari[etichetaPrincipala], exemplu)
    else:
        return -1

dateValidare = pd.read_csv(caleFisierValidare)
XValidare = dateValidare.iloc[:, 1:-1].values.astype(float)
yValidare = dateValidare.iloc[:, -1].values

def codificareEticheta(etichetaReala):
    if 200 <= etichetaReala < 800:
        return codificaSubinterval0
    elif 800 <= etichetaReala < 1400:
        return codificaSubinterval1
    elif etichetaReala >= 1400:
        return codificaSubinterval2
    else:
        return -1

corecte = 0
for exemplu, etichetaReal in zip(XValidare, yValidare):
    predicte = preziceFinal(arborePrincipal, arboriSecundari, exemplu)
    funcCodificare = codificareEticheta(etichetaReal)
    if funcCodificare != -1:
        subintervalCodificat = funcCodificare(etichetaReal)
        if predicte == subintervalCodificat:
            corecte += 1

rataEroare = 1 - (corecte / len(yValidare))
print(f"Rata de eroare: {rataEroare:.4f} ({corecte}/{len(yValidare)} corecte)")
