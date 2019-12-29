import pandas as pd
from timeseries import get_series, test_stationarity, seasonal, get_d, test_residual

df = pd.read_csv('../files/outputs/t_b_ts.csv')

ts = get_series(df, objective='total')

# Analysis
test_stationarity(ts)  # test stationarity
seasonal(ts)  # extract trend and seasonality
get_d(ts)  # order of integration
# first difference
ts_d = ts.diff().dropna()
test_stationarity(ts_d, model='FirstDiff')  # likely stationary


