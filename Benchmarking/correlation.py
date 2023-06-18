import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Fixing random state for reproducibility
np.random.seed(19680801)

arr = pd.read_csv('~/Downloads/rnd/2013-7/{}.csv'.format(1), sep=';	', engine='python',
                      usecols=['CPU usage [%]']).to_numpy().flatten()

print(arr.shape)
# t = [*range(0,arr.shape[0],1)]
t = np.arange(0,arr.shape[0])
print(t)
print(len(t))
# x, y = np.random.randn(2, 100)
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
# ax1.xcorr(arr, t, usevlines=True, maxlags=50, normed=True, lw=2)
# ax1.grid(True)
#
# ax2.acorr(arr, usevlines=True, normed=True, maxlags=50, lw=2)
# ax2.grid(True)
plot_pacf(arr, auto_ylims=True, ax=ax2, lags=100)
plot_acf(arr, auto_ylims=True, ax=ax1 , lags=100)
plt.show()