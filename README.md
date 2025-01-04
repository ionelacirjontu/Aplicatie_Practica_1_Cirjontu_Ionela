# Aplicatie_Practica_1_Cirjontu_Ionela

Acesta este un algoritm de arbore de decizie implementat în Python, folosind bibliotecile NumPy și Pandas pentru procesarea datelor. Algoritmul incearca prezicerea soldului din luna decembrie antrenand modelul cu date preluate pentru luna noiembrie.


### 1. Importarea bibliotecilor

```
import numpy as np
import pandas as pd
```
NumPy este folosit pentru manipularea si calculul numeric al datelor.
Pandas este folosit pentru manipularea datelor intr-un format tabular, similar cu tabelele de baze de date.

### 2. Definirea clasei nod
```
class Nod:
    def __init__(self, atribut=None, prag=None, eticheta=None):
        self.atribut = atribut
        self.prag = prag
        self.eticheta = eticheta
        self.stanga = None
        self.dreapta = None
```
Clasa Nod reprezintă un nod al arborelui de decizie.Campurile au semnificatia:
atribut: atributul pe care se face impartirea din nodul respectiv.
prag: pragul folosit pentru a imparti setul de date.
eticheta: eticheta asociata nodului.
stanga si dreapta: subarborele stang si drept al nodului respectiv.

### 3.Calcularea entropiei
```
def entropie(y):
    valori, numar_ocurrente = np.unique(y, return_counts=True)
    probabilitati = numar_ocurrente / len(y)
    return -np.sum(probabilitati * np.log2(probabilitati))
```
Calculează entropia unui set de etichete y.

### 4.Castigul de informatie
```
def castigInformatie(y, yStanga, yDreapta):
    return entropie(y) - (len(yStanga) / len(y) * entropie(yStanga) + len(yDreapta) / len(y) * entropie(yDreapta))
```

Calculeaza castigul de informatie pentru impartirea datelor in subarborele yStanga si subarborele yDreapta

### 5.Gasirea celui mai bun atribut si prag 
```
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
```
Calculeaza cel mai bun atribut la un moment dat si pragul cel mai semnificativ pentru acesta


### 6.Construirea propriu-zisa a arborelui 
```
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
```
Construieste un arbore de decizie pe baza unui set de date de intrare X si a etichetelor y. Aceasta functioneaza recursiv pentru a imparti datele în subgrupuri pe masura ce arborele se dezvolta, folosind cea mai buna impartire, folosind functia anterioara la fiecare pas.

### 7. Prezicerea exemplelor de validare in arborele principal
```
def prezice(arbore, X):
    if arbore.eticheta is not None:
        return arbore.eticheta
    if X[arbore.atribut] <= arbore.prag:
        return prezice(arbore.stanga, X)
    else:
        return prezice(arbore.dreapta, X)
```
Parcurge recursiv arborele pana ajunge la eticheta corespunzatoare instantei curente

### 8.Codificare sold in arborele principal
```
def codificaSold(sold):
    if 200 <= sold < 800:
        return 0
    elif 800 <= sold < 1400:
        return 1
    elif sold >= 1400:
        return 2
    else:
        return -1
```
Codificarea valorilor soldului in intervale 

### 9.Codificare sold subarbore 1
```
def codificaSubinterval0(sold):
    if 200 <= sold < 400:
        return 3
    elif 400 <= sold < 600:
        return 4
    elif 600 <= sold < 800:
        return 5
    else:
        return -1
```

### 10.Codificare sold subarbore 2
```
def codificaSubinterval1(sold):
    if 800 <= sold < 1000:
        return 6
    elif 1000 <= sold < 1200:
        return 7
    elif 1200 <= sold < 1400:
        return 8
    else:
        return -1
```

### 11.Codificare sold subarbore 3
```
def codificaSubinterval2(sold):
    if 1400 <= sold < 1800:
        return 9
    elif 1800 <= sold < 2200:
        return 10
    elif sold >= 2200:
        return 11
    else:
        return -1
```

### 12.Incarcare set antrenament si selectarea atributelor relevate, 1 si 3
```
caleFisierAntrenament = 'dateAntrenament.csv'
caleFisierValidare = 'dateValidare.csv'
atributeSelectate = [1, 3] 
dateAntrenament = pd.read_csv(caleFisierAntrenament)
XAntrenament = dateAntrenament.iloc[:, atributeSelectate].values.astype(float)  
yAntrenament = np.array([codificaSold(s) for s in dateAntrenament.iloc[:, -1].values])
```

### 13.Creare arbore principal
```
arborePrincipal = construiesteArbore(XAntrenament, yAntrenament)
```
Antrenarea arborelui cu setul de antrenament
### 14.Crearea arborilor secundari 
```
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
```
Antrenarea arborilor secundari cu instantele de antrenament corespunzatoare indicelui subarborelui.Daca in arborele principal i se pune eticheta 0, aceasta instanta se foloseste la antrenarea subarborelui 0

### 15.Prezicere finala
```
def preziceFinal(arborePrincipal, arboriSecundari, exemplu):
    etichetaPrincipala = prezice(arborePrincipal, exemplu)
    if etichetaPrincipala in [0, 1, 2]:
        return prezice(arboriSecundari[etichetaPrincipala], exemplu)
    else:
        return -1
```
Eticheta finala oferita de subarborele corespunzator

### 16.Incarcarea setului de validare 
```
dateValidare = pd.read_csv(caleFisierValidare)
XValidare = dateValidare.iloc[:, atributeSelectate].values.astype(float)  
yValidare = dateValidare.iloc[:, -1].values
```

### 17.Codificarea reala a instantei din setul de validare
```
def codificareEticheta(etichetaReala):
    if 200 <= etichetaReala < 800:
        return codificaSubinterval0
    elif 800 <= etichetaReala < 1400:
        return codificaSubinterval1
    elif etichetaReala >= 1400:
        return codificaSubinterval2
    else:
        return -1
```

### 18.Parcurgerea setului de validare si clasificarea lui folosind arborele principal si subarborii si contorizarea prezicerilor corecte
```
corecte = 0
for exemplu, etichetaReal in zip(XValidare, yValidare):
    predicte = preziceFinal(arborePrincipal, arboriSecundari, exemplu)
    funcCodificare = codificareEticheta(etichetaReal)
    if funcCodificare != -1:
        subintervalCodificat = funcCodificare(etichetaReal)
        if predicte == subintervalCodificat:
            corecte += 1
```
### 19.Rata eroare
```
rataEroare = 1 - (corecte / len(yValidare))
```



