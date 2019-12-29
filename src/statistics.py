""" compute the average, std, median and percentile of the bookings/parkings duration for the 2017/10"""

from time import time
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from connection import Connection
from pipeline_ict4ts import pip_d

client = Connection()
db = client.db

start = datetime.timestamp(datetime(2017, 10, 1))
end = datetime.timestamp(datetime(2017, 10, 31, 23, 59))

start_ = time()
to_b_d_month = list(db['PermanentBookings'].aggregate(pip_d(start_timestamp=start, end_timestamp=end)))
to_p_d_month = list(db['PermanentParkings'].aggregate(pip_d(start_timestamp=start, end_timestamp=end)))

po_p_d_month = list(db['PermanentParkings'].aggregate(pip_d(start_timestamp=start, end_timestamp=end, city='Portland')))
po_b_d_month = list(db['PermanentBookings'].aggregate(pip_d(start_timestamp=start, end_timestamp=end, city='Portland')))

eto_b_d_month = list(db['enjoy_PermanentBookings'].aggregate(pip_d(start_timestamp=start, end_timestamp=end)))
eto_p_d_month = list(db['enjoy_PermanentParkings'].aggregate(pip_d(start_timestamp=start, end_timestamp=end)))
end_ = time()

elap = end_ - start_
print(f"Time elapsed: {elap}")

df_b_d_month_t = pd.DataFrame(to_b_d_month)
df_p_d_month_t = pd.DataFrame(to_p_d_month)

df_b_d_month_te = pd.DataFrame(eto_b_d_month)
df_p_d_month_te = pd.DataFrame(eto_p_d_month)

df_b_d_month_p = pd.DataFrame(po_b_d_month)
df_p_d_month_p = pd.DataFrame(po_p_d_month)

df_b_d_month_p.to_csv('b_d_month_p.csv')
df_p_d_month_p.to_csv('p_d_month_p.csv')

df_b_d_month_t.to_csv('b_d_month_t.csv')
df_p_d_month_t.to_csv('p_d_month_t.csv')

df_b_d_month_te.to_csv('b_d_month_te.csv')
df_p_d_month_te.to_csv('p_d_month_te.csv')

# ========================
# Bookings
# =======================
# avg
avg_t_b = [x/60 for x in df_b_d_month_t['AvgDuration']]
avg_te_b = [x/60 for x in df_b_d_month_te['AvgDuration']]
avg_p_b = [x/60 for x in df_b_d_month_p['AvgDuration']]
# std
std_t_b = [x/60 for x in df_b_d_month_t['StdDuration']]
std_te_b = [x/60 for x in df_b_d_month_te['StdDuration']]
std_p_b = [x/60 for x in df_b_d_month_p['StdDuration']]
# median
to_median = []
eto_media = []
po_media = []
# percentile
to_per = []
eto_per = []
po_per = []

travel_to = df_b_d_month_t['travel']
travel_toe = df_b_d_month_te['travel']
travel_p =  df_b_d_month_p['travel']

for e in travel_to:
    ls = [e[i]/60 for i in range(len(e))]
    array = np.asarray(ls)
    to_median.append(np.median(array))
    to_per.append(np.percentile(array, 75))

for e in travel_toe:
    ls = [e[i]/60 for i in range(len(e))]
    array = np.asarray(ls)
    eto_media.append(np.median(array))
    eto_per.append(np.percentile(array, 75))

for e in travel_p:
    ls = [e[i]/60 for i in range(len(e))]
    array = np.asarray(ls)
    po_media.append(np.median(array))
    po_per.append(np.percentile(array, 75))

# =====================
# Plots Bookings
# =====================
#plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
date = df_b_d_month_t['date'].astype('O')
#df = df.set_index('date')

fig.suptitle('Statistics Bookings')
ax1.plot(date, avg_t_b, linewidth=1, marker='.', label='AVG')
ax1.plot(date, std_t_b, linewidth=1, marker='^', label='STD')
ax1.plot(date, to_median, linewidth=1, marker='*', label='Median')
ax1.plot(date, to_per, linewidth=1, marker='8', label='P(75%)')
#ax1.ylabel('duration time [s]')
fig.autofmt_xdate()
# use a more precise date string for the x axis locations in the
# toolbar
ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax1.set_title('Turin')
ax1.set_ylabel('time [min]')
ax1.set_ylim(3, 60)
#ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           #ncol=4, mode="expand", borderaxespad=0.)
ax1.grid(linestyle='--', linewidth=.4, which="both")


#plt.subplot(312)

date = df_b_d_month_te['date'].astype('O')
#df = df.set_index('date')
ax2.plot(date, avg_te_b, linewidth=1, marker='.', label='AVG')
ax2.plot(date, std_te_b, linewidth=1, marker='^', label='STD')
ax2.plot(date, eto_media, linewidth=1, marker='*', label='Median')
ax2.plot(date, eto_per, linewidth=1, marker='8', label='P(75%)')

#ax2.ylabel('duration time [s]')

# use a more precise date string for the x axis locations in the
# toolbar
ax2.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax2.set_title('Turin Enjoy')
#fig.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
          # ncol=4, mode="expand", borderaxespad=0.)
ax2.grid(linestyle='--', linewidth=.4, which="both")
ax2.set_ylim(3, 60)

#plt.subplot(313)
date = df_b_d_month_p['date'].astype('O')
#df = df.set_index('date')
ax3.plot(date, avg_p_b, linewidth=1, marker='.', label='AVG')
ax3.plot(date, std_p_b, linewidth=1, marker='^', label='STD')
ax3.plot(date, po_media, linewidth=1, marker='*', label='Median')
ax3.plot(date, po_per, linewidth=1, marker='8', label='P(75%)')

#ax3.ylabel('duration time [s]')

# use a more precise date string for the x axis locations in the
# toolbar
ax3.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax3.set_title('Portland')
ax3.legend(bbox_to_anchor=(1.05, 4.), loc='upper left',
          borderaxespad=0.)
ax3.grid(linestyle='--', linewidth=.4, which="both")
ax3.set_ylim(3, 60)
plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9, hspace=1)
plt.savefig(fname="plots/Book_stats.png")
plt.close()

# ========================
# Parkings
# =======================

# avg
avg_t_p = [x/60 for x in df_p_d_month_t['AvgDuration']]
avg_te_p = [x/60 for x in df_p_d_month_te['AvgDuration']]
avg_p_p = [x/60 for x in df_p_d_month_p['AvgDuration']]
# std
std_t_p = [x/60 for x in df_p_d_month_t['StdDuration']]
std_te_p = [x/60 for x in df_p_d_month_te['StdDuration']]
std_p_p = [x/60 for x in df_p_d_month_p['StdDuration']]
# median
to_median = []
eto_media = []
po_media = []
# percentile
to_per = []
eto_per = []
po_per = []

travel_to = df_p_d_month_t['travel']
travel_toe = df_p_d_month_te['travel']
travel_p = df_p_d_month_p['travel']

for e in travel_to:
    ls = [e[i]/60 for i in range(len(e))]
    array = np.asarray(ls)
    to_median.append(np.median(array))
    to_per.append(np.percentile(array, 75))

for e in travel_toe:
    ls = [e[i]/60 for i in range(len(e))]
    array = np.asarray(ls)
    eto_media.append(np.median(array))
    eto_per.append(np.percentile(array, 75))

for e in travel_p:
    ls = [e[i]/60 for i in range(len(e))]
    array = np.asarray(ls)
    po_media.append(np.median(array))
    po_per.append(np.percentile(array, 75))

# =====================
# Plots parkings
# =====================
#plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
date = df_p_d_month_t['date'].astype('O')
#df = df.set_index('date')
fig.suptitle('Statistics Parkings')
ax1.plot(date, avg_t_p, linewidth=1, marker='.', label='AVG')
ax1.plot(date, std_t_p, linewidth=1, marker='^', label='STD')
ax1.plot(date, to_median, linewidth=1, marker='*', label='Median')
ax1.plot(date, to_per, linewidth=1, marker='8', label='P(75%)')
ax1.set_ylabel('time [min]')
#ax1.ylabel('duration time [s]')
fig.autofmt_xdate()
# use a more precise date string for the x axis locations in the
# toolbar
ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax1.set_title('Turin')
ax1.set_ylim(3, 80)
ax1.grid(linestyle='--', linewidth=.4, which="both")

#plt.subplot(312)
date = df_p_d_month_te['date'].astype('O')
#df = df.set_index('date')
ax2.plot(date, avg_te_p, linewidth=1, marker='.', label='AVG')
ax2.plot(date, std_te_p, linewidth=1, marker='^', label='STD')
ax2.plot(date, eto_media, linewidth=1, marker='*', label='Median')
ax2.plot(date, eto_per, linewidth=1, marker='8', label='P(75%)')
ax2.set_ylim(3, 80)
#plt.ylabel('duration time [s]')
#fig.autofmt_xdate()
# use a more precise date string for the x axis locations in the
# toolbar
ax2.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax2.set_title('Turin Enjoy')

ax2.grid(linestyle='--', linewidth=.4, which="both")

#plt.subplot(313)
date = df_p_d_month_p['date'].astype('O')
#df = df.set_index('date')
ax3.plot(date, avg_p_p, linewidth=1, marker='.', label='AVG')
ax3.plot(date, std_p_p, linewidth=1, marker='^', label='STD')
ax3.plot(date, po_media, linewidth=1, marker='*', label='Median')
ax3.plot(date, po_per, linewidth=1, marker='8', label='P(75%)')
ax3.set_ylim(3, 80)
#plt.ylabel('duration time [s]')
#plt.autofmt_xdate()
# use a more precise date string for the x axis locations in the
# toolbar
ax3.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
ax3.set_title('Portland')

ax3.legend(bbox_to_anchor=(1.05, 4.), loc='upper left',
          borderaxespad=0.)
ax3.grid(linestyle='--', linewidth=.4, which="both")
plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9, hspace=1)
plt.savefig(fname="plots/Park_stats.png")
plt.close()






