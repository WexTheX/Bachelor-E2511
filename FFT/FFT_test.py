from scipy.fft import fft, ifft
import numpy as np
import pandas as pd

df = pd.read_csv('Muse_Test_File.txt', delimiter='\t')
df.to_csv('Muse_Test_File.csv', index = None)