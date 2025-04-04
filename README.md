# Bachelor-E2511
Code for bachelor project for group E2511, NTNU

Eksempel med en "GRIN" fil:
.txt filer må legges inn manuelt i Preprocessing/Datafiles/Grinding og renames til "GRIND_n" 

antall .txt filer i hver Datafiles mappe blir telt, og navnet blir lagt inn i listen "sets" som en streng. "setsLabel" får

# TODO 

1) Separer typer uten å måtte ha en folder for hver (unngå dobbelt opp med filer) [Low pri]
2) Implementer K-nearest og Naive-Bayers, Gradient boosting
3) Skisser boundary conditions i SVM [Problem med 2d PCA] [ferdig]
4) Legge RF inn i machineLearning.py [Ferdig]
5) Kode som tar inn test som logger aktiviteter
6) Lage en downsampler [Ferdig]
    6.1) Implementer downsampler naturlig or something

...

100) Sanntidslogging

...

1000) CNN



Hva forventer vi å bli ferdig med til påske?
13/04

1) Skrive i metode-bit hvordan vi prosesserer data før vi sender inn i maskinlæring 

2) Sette opp pipeline for å teste filer inne i testFiles 
2.1) Logge resultatet, lagre resultatet i en fil

3) Få til Real-Time
3.2) Classifier og PCA må packages gjennom Pickle
3.3) Funksjon som appender hver IMU-linje til f.eks en liste eller df (Kan denne kjøre på egen kjerne? Og classifier på en annen?)

% Pseudo

def saveWindow():
    if len(list) < window_length_seconds * fs: 
            list.append(new_IMU_row)

        else:
            window = list
            list = []
            return window

4) Ordentlig analyse av hvilken features er nødvendig (også rapportmat, dette må begrunnes)

5) Bedre grunnlag på hvor gode classifiers er (ROC AUC, log loss for enkelte classifiers)

6) Flere bilder / grafer til rapporten


    Endelig confusion matrix
    plots av markante features (feks magnometer for sveising, accel fra vinkelsliper)
    vis frem features som vektlegges mest, og forklar hvorfor
    KNN og SVM visualisering (boundary conditions og områder)

    Kollasj med alle forskjellige classifiers

Extra: 

7) Se på flere features (moving average ...)
    Finn fine grafer til rapporten