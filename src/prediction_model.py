from timeseries import get_series, get_d, test_stationarity, arimamodel, plot_acf_pacf, seasonal, autosarima, autoarima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


#(1,1,2)(2,1,2,12) aic=5387.9134

df = pd.read_csv('../files/outputs/t_b_ts.csv')

ts = get_series(df, objective='total')

# data splitted
N_train = int(ts.shape[0]*0.6)
x_train = ts[0:N_train]
x_test = ts[N_train+1:]





dict_, pdq = autoarima(ts)

"""
mod_s = sm.tsa.statespace.SARIMAX(x_train,
                                order=(1, 1, 2),
                                seasonal_order=(2, 1,2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                disp=False)

results = mod_s.fit()
print(results.summary())


yhat = results.get_prediction(start=N_train-12)
forecast = results.get_forecast(steps=12)



fig, ax = plt.subplots(figsize=(10,4))

# Plot the results
x_train.plot(ax=ax, style='k.', label='Observations')
yhat.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')
predict_ci = yhat.conf_int(alpha=0.05)
predict_index = np.arange(len(predict_ci))
ax.fill_between(predict_index[2:], predict_ci.iloc[2:, 0], predict_ci.iloc[2:, 1], alpha=0.1)

forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
forecast_ci = forecast.conf_int()
forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.1)

# Cleanup the image
ax.set_ylim((4, 8))
legend = ax.legend(loc='lower left')
"""











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

