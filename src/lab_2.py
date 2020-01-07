"""ICT4TS LAB 2: build a model able to predict future rentals in time by exploiting the historically data i.e gives the
number of rental in the past predict the number of rental in the future.
First step: check if and under what conditions the time series are stationary
Second step: compute the ACF and/or PACF for determining the grade of the lag
Third Step: build the right ARIMA model for the data.
Forty Step: test the model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from connection import Connection
from pipeline_ict4ts import pip_uot, pip_uotF, pip_hmt, pip_d
from timeseries import test_stationarity, test_residual, get_series, get_d


client = Connection()
db = client.db

start = datetime.timestamp(datetime(2017, 10, 1))
end = datetime.timestamp(datetime(2017, 11, 1))


to_b_uotF = list(db['PermanentBookings'].aggregate(pip_uotF(start_timestamp=start, end_timestamp=end,
                                                                 objective='year')))


eto_b_uotF = list(db['enjoy_PermanentBookings'].aggregate(pip_uotF(start_timestamp=start, end_timestamp=end,
                                                                        objective='year')))


po_b_uotF = list(db['PermanentBookings'].aggregate(pip_uotF(city='Portland', start_timestamp=start,
                                                                 end_timestamp=end,
                                                                objective='year')))

ts_b_t = get_series(to_b_uotF, objective='total')
ts_b_te = get_series(eto_b_uotF, objective='total')
ts_b_po = get_series(po_b_uotF, objective='total')

df_t = pd.DataFrame(ts_b_t)
df_et = pd.DataFrame(ts_b_te)
df_po = pd.DataFrame(ts_b_po)

df_t.to_csv('../files/outputs/t_b_ts.csv')
df_et.to_csv('../files/outputs/te_b_ts.csv')
df_po.to_csv('../files/outputs/po_b_ts.csv')


















