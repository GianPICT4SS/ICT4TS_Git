from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
from datetime import datetime
import logging
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import normaltest
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()




def test_stationarity(timeseries, collection='PermanentBookings', model='Original', cutoff=0.001):

    # Determing rolling statistics

    rolmean = timeseries.rolling(3).mean()
    rolstd = timeseries.rolling(3).std()

    # Plot rolling statistics:
    fig, ax1 = plt.subplots()
    ax1.plot(timeseries.index.values, timeseries, label='Observed')
    ax1.plot(timeseries.index.values, rolmean, label='Roll Mean')
    ax1.plot(timeseries.index.values, rolstd, label='Roll std')
    ax1.set_title(f'{collection}: Rolling Mean & Standard Deviation of {model}')
    ax1.legend(loc='best')
    ax1.grid(linestyle='--', linewidth=.4, which="both")

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    # use a more precise date string for the x axis locations in the
    # toolbar
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    plt.savefig(f'../plots/rolling_{collection}_{model}.png')
    #plt.show(block=False)
    plt.close()

    # Perform Dickey-Fuller test:
    print(f'{collection}: Results of Dickey-Fuller Test for {model}:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value {key}'] = value
    pvalue = dfoutput[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    print(dfoutput)
    dfoutput = str(dfoutput)
    file = open(f'../files/ts_analysys_{collection}_{model}.txt', 'w')
    file.write(dfoutput)
    file.close()


def seasonal(ts, freq=1, collection='PermamentBookings'):

    result = seasonal_decompose(ts, freq=freq)
    fig = plt.figure()
    fig = result.plot()

    plt.savefig(f'../plots/{collection}.png')


def get_series(data_ls, objective):

    df = pd.DataFrame(data_ls)
    try:
        df = df.set_index('date')
        df = df.set_index(pd.to_datetime(df.index))
        df = df.resample('D').mean()
        df.dropna(inplace=True)
        ts = df[objective]
        return ts

    except ValueError as e:
        print(f'{e}')
        pass






def heatmap(X, columns, name):

        df = pd.DataFrame(X, columns=columns)
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        plt.title(name)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

        fname = name + '.png'
        plt.savefig('../plots/' + fname)
        pass

def get_d(ts):
    # ACF and PACF analysis
    # find the number of differencing d
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(311)
    fig = plot_acf(ts, ax=ax1, lags=50, title='ACF for the original data')
    ax2 = fig.add_subplot(312)
    fig = plot_acf(ts.diff().dropna(), ax=ax2, lags=50, title='First difference')
    ax3 = fig.add_subplot(313)
    fig = plot_acf(ts.diff().diff().dropna(), ax=ax3, lags=50, title='Second difference')

def arimamodel(timeseries):

    automodel = pm.auto_arima(timeseries,
                              start_p=0,
                              start_q=0,

                              trace=True)
    return automodel


def test_residual(results_ts, model='MA'):

    resid = results_ts.resid
    fig = plt.figure(figsize=(12, 8))
    ax0 = fig.add_subplot(111)
    sns.distplot(resid, fit=stats.norm, ax=ax0)  # need to import scipy.stats
    # Get the fitted parameters used by the function
    (mu, sigma) = stats.norm.fit(resid)
    # Now plot the distribution using
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title(f'Residual distribution of {model}')
    # ACF and PACF
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(results_ts.resid, lags=40, ax=ax1, title=f'ACF of {model}')
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(results_ts.resid, lags=40, ax=ax2, title=f'PACF of {model}')



if __name__ == "__main__":

    df = pd.read_csv('uotB_year.csv')
    df = df.drop(['Unnamed: 0', '_id'], axis=1)
    df = df.set_index('date')
    ts = df['total']
    ts_1d = ts.diff().dropna()
    ts_2d = ts_1d.diff().dropna()
    ts_3s = ts_2d.diff().dropna()
    #seasonal(ts)


    # ACF
   # get_d(ts)  # d = 1
   # test_stationarity(ts, model='Original')
   # test_stationarity(ts_1d, model='First_difference')

   # plot_acf(ts_1d, lags=60)
   # plot_pacf(ts_1d, lags=60)


    model_ma2 = ARIMA(ts, order=(0, 1, 2))
    model_arima = ARIMA(ts, order=(1, 1, 2))
    model_arima_712 = ARIMA(ts, order=(2, 1, 2))
    results_ma2 = model_ma2.fit()
    results_arima = model_arima.fit()
    results_arima_712 = model_arima_712.fit()
    results_ma2.plot_predict(210, 365)
    results_arima.plot_predict(210, 365)
    results_arima_712.plot_predict(210, 365)

    automodel = arimamodel(ts)

    #test_residual(results_ma1, model='ARIMA_011')
    #test_residual(results_arima, model='ARIMA_112')
    #test_residual(results_arima_211, model='ARIMA_212')
   # test_residual(automodel, model='SARIMAX_100')



