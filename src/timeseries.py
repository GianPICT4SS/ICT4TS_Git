from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm


import seaborn as sns
from scipy import stats
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'
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
        df = df.resample('H').mean()
        df.dropna(inplace=True)
        ts = df[objective]
        return ts

    except ValueError as e:
        print(f'{e}')
        pass


def plot_acf_pacf(ts, lags=10):

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(ts, ax=ax1, lags=lags)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(ts, ax=ax2, lags=lags)


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

def get_d(ts, lags=10):
    # ACF and PACF analysis
    # find the number of differencing d
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(311)
    fig = plot_acf(ts, ax=ax1, lags=lags, title='ACF for the original data')
    ax2 = fig.add_subplot(312)
    fig = plot_acf(ts.diff().dropna(), ax=ax2, lags=lags, title='First difference')
    ax3 = fig.add_subplot(313)
    fig = plot_acf(ts.diff().diff().dropna(), ax=ax3, lags=lags, title='Second difference')

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
    fig = plot_acf(results_ts.resid, lags=15, ax=ax1, title=f'ACF of {model}')
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(results_ts.resid, lags=15, ax=ax2, title=f'PACF of {model}')

def autosarima(ts, order=3):

    p = d = q = range(0, order)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    dic_aic = {}
    print('SARIMA tunning starts:')
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                dic_aic[param, param_seasonal] = results.aic

                print('ARIMA{}x{} - AIC:{}'.format(param,
                                                     param_seasonal,
                                                     results.aic))
            except:
                continue

    return dic_aic









