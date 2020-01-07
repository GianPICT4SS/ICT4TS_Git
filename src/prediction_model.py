from timeseries import get_series, get_d, test_stationarity, arimamodel, plot_acf_pacf, seasonal, autosarima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df = pd.read_csv('../files/outputs/t_b_ts.csv')

ts = get_series(df, objective='total')

# data splitted
N_train = int(ts.shape[0]*0.6)
x_train = ts[0:N_train]
x_test = ts[N_train+1:]

model = arimamodel(ts)

model = model.fit(x_train)

y_test_pred = model.predict(5)

fig, ax = plt.subplots()

ax.plot(ts.index.values, ts)
ax.plot(y_test_pred)



"""
# Analysis
test_stationarity(ts)  # test stationarity
#get d
get_d(ts, lags=40)
# first difference
ts_d = ts.diff().dropna()
test_stationarity(ts_d, model='FirstDiff')  # likely stationary
plot_acf_pacf(ts_d, lags=40)
seasonal(ts)
automodel = arimamodel(ts)
"""

#dict_ = autosarima(ts)