import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from connection import Connection
from pipeline_ict4ts import pip_d

client = Connection()

db = client.db

start = datetime.timestamp(datetime(2017, 10, 1))
end = datetime.timestamp(datetime(2017, 10, 31))

duration_B_T = list(db['PermanentBookings'].aggregate(pip_d(start, end), allowDiskUse=True))
duration_B_P = list(db['PermanentBookings'].aggregate(pip_d(start, end, city='Portland'), allowDiskUse=True))
duration_B_Te = list(db['enjoy_PermanentBookings'].aggregate(pip_d(start, end, city='Torino'), allowDiskUse=True))

duration_P_T = list(db['PermanentParkings'].aggregate(pip_d(start, end, city='Torino'), allowDiskUse=True))
duration_P_P = list(db['PermanentParkings'].aggregate(pip_d(start, end, city='Portland'), allowDiskUse=True))
duration_P_Te = list(db['enjoy_PermanentParkings'].aggregate(pip_d(start, end, city='Torino'), allowDiskUse=True))

# time booking
time_B_T = [(car['duration']) for car in duration_B_T]
time_B_P = [(car['duration']) for car in duration_B_P]
time_B_Te = [(car['duration']) for car in duration_B_Te]
# time parking
time_P_T = [(car['duration']) for car in duration_P_T]
time_P_P = [(car['duration']) for car in duration_P_P]
time_P_Te = [(car['duration']) for car in duration_P_Te]

# CDF
# duration, frequency_duration = zip(*Counter(dict_duration.values()).items())
#booking
time_B_T = [x/60 for x in time_B_T]
time_B_P = [x/60 for x in time_B_P]
time_B_Te = [x/60 for x in time_B_Te]
#parking
time_P_T = [x/60 for x in time_P_T]
time_P_P = [x/60 for x in time_P_P]
time_P_Te = [x/60 for x in time_P_Te]

# x, y PermanentBookings
time_B_T = np.sort(np.asarray(time_B_T))
y_B_T = np.arange(len(time_B_T))/float(len(time_B_T))
time_B_P = np.sort(np.asarray(time_B_P))
y_B_P = np.arange(len(time_B_P))/float(len(time_B_P))
time_B_Te = np.sort(np.asarray(time_B_Te))
y_B_Te = np.arange(len(time_B_Te))/float(len(time_B_Te))

# x, y PermanentParkings
time_P_T = np.sort(np.asarray(time_P_T))
y_P_T = np.arange(len(time_P_T))/float(len(time_P_T))
time_P_P = np.sort(np.asarray(time_P_P))
y_P_P = np.arange(len(time_P_P))/float(len(time_P_P))
time_P_Te = np.sort(np.asarray(time_P_Te))
y_P_Te = np.arange(len(time_P_Te))/float(len(time_P_Te))

# plot ECDF Permanent Bookings

plt.subplot(211)
plt.plot(time_B_T, y_B_T, linewidth=1, label='Torino')
plt.plot(time_B_P, y_B_P, linewidth=1, label='Portland')
plt.plot(time_B_Te, y_B_Te, linewidth=1, label='Torino enjoy')

plt.xlabel("Time [min]")
plt.ylabel("ECDF")
plt.legend(loc='lower right')
plt.title("ECDF Permanent Bookings")
plt.xscale("log")
plt.grid(linestyle='--', linewidth=.4, which="both")
#plt.savefig(fname='plots/ECDF_PBookings.png')
#plt.show()

# Plot ECDF Permanent Parkings
plt.subplot(212)
plt.plot(time_P_T, y_P_T, linewidth=1, label='Torino')
plt.plot(time_P_P, y_P_P, linewidth=1, label='Portland')
plt.plot(time_P_Te, y_P_Te, linewidth=1, label='Torino enjoy')
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=1)
#plt.axis([0, 500, 0, 1])
plt.xlabel("Time [min]")
plt.ylabel("ECDF")
plt.legend(loc='lower right')
plt.title("ECDF Permanent Parkings")
plt.xscale("log")
plt.grid(linestyle='--', linewidth=.4, which="both")
plt.savefig(fname='../plots/ECDF_P_Bookings_Parkings_F.png')

plt.close()
