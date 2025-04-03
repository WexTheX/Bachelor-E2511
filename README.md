# Bachelor-E2511
Code for bachelor project for group E2511, NTNU

Eksempel med en "GRIN" fil:
.txt filer må legges inn manuelt i Preprocessing/Datafiles/Grinding og renames til "GRIND_n" 

antall .txt filer i hver Datafiles mappe blir telt, og navnet blir lagt inn i listen "sets" som en streng. "setsLabel" får

# TODO 

1) Separer typer uten å måtte ha en folder for hver (unngå dobbelt opp med filer)
2) Implementer K-nearest og Naive-Bayers, Gradient boosting
3) Skisser boundary conditions i SVM [Problem med 2d PCA] [ferdig]
4) Legge RF inn i machineLearning.py [Ferdig]
5) Kode som tar inn test som logger aktiviteter
6) Lage en downsampler

...

100) Sanntidslogging

...

1000) CNN


old want_plots
# for i in range(1, len(variables)):
    #     plotWelch(sets[0], variables[i], Fs, False)
    #     plotWelch(sets[0], variables[i], Fs, True)
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Power Spectral Density')
    #     plt.title('Welch PSD, %s' % variables[i])
    #     plt.grid()
    #     plt.figure()