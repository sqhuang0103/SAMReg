import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="darkgrid")
time = np.arange(0, 2*np.pi, 0.1)
sin_waves = np.sin(time)
sin_waves = np.expand_dims(sin_waves, axis=-1)
noise = np.random.random((time.size, 10)) - 0.5
data = sin_waves + noise
data_mean = np.mean(data, axis=1)
data_std = np.std(data, axis=1)
data_var = np.var(data, axis=1)
data_max = np.max(data, axis=1)
data_min = np.min(data, axis=1)
plt.figure()
print(time)
plt.plot(time, data_mean,marker='o', color='deeppink', label='mean')
plt.plot(time, sin_waves, 'b-', label='ideal')
plt.fill_between(time, data_mean - data_std, data_mean + data_std, color='violet', alpha=0.2)
plt.legend()

plt.show()