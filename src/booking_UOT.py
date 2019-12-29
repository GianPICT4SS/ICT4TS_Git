""" Booking Utilization Over Time: consider the system utilization over time: aggregate rentals per hour of the day,
and then plot the number of booked cars (or percentage of booked/parked cars) per hour versus time of day."""

from pipeline_ict4ts import pip_uot, pip_uotF
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from connection import Connection


client = Connection()
db = client.db


start = datetime.timestamp(datetime(2017, 10, 1))
end = datetime.timestamp(datetime(2017, 10, 31))

# =======================================================
# BOOKINGS PER HOUR in WEEK (Utilization Over Time)
# =======================================================
# No Filter
# Bookings

to_b = list(db['PermanentBookings'].aggregate(pip_uot(city='Torino', start_timestamp=start, end_timestamp=end)))
po_b = list(db['PermanentBookings'].aggregate(pip_uot(city='Portland', start_timestamp=start, end_timestamp=end)))
to_be = list(db['enjoy_PermanentBookings'].aggregate(pip_uot(city='Torino', start_timestamp=start, end_timestamp=end)))

to_b_x = np.arange(len(to_b))
po_b_x = np.arange(len(po_b))
to_be_x = np.arange(len(to_be))

to_b_y = [car['total'] for car in to_b]
po_b_y = [car['total'] for car in po_b]
to_be_y = [car['total'] for car in to_be]

to_b_y = np.asarray(to_b_y)
po_b_y = np.asarray(po_b_y)
to_be_y = np.asarray(to_be_y)
# Parkings
to_p = list(db['PermanentParkings'].aggregate(pip_uot(city='Torino', start_timestamp=start, end_timestamp=end)))
po_p = list(db['PermanentParkings'].aggregate(pip_uot(city='Portland', start_timestamp=start, end_timestamp=end)))
to_pe = list(db['enjoy_PermanentParkings'].aggregate(pip_uot(city='Torino', start_timestamp=start, end_timestamp=end)))

to_p_x = np.arange(len(to_p))
po_p_x = np.arange(len(po_p))
to_pe_x = np.arange(len(to_pe))

to_p_y = [car['total'] for car in to_p]
po_p_y = [car['total'] for car in po_p]
to_pe_y = [car['total'] for car in to_pe]

to_p_y = np.asarray(to_p_y)
po_p_y = np.asarray(po_p_y)
to_pe_y = np.asarray(to_pe_y)
# Filter
# Bookings
to_bF = list(db['PermanentBookings'].aggregate(pip_uotF(city='Torino', start_timestamp=start, end_timestamp=end)))
po_bF = list(db['PermanentBookings'].aggregate(pip_uotF(city='Portland', start_timestamp=start, end_timestamp=end)))
to_beF = list(db['enjoy_PermanentBookings'].aggregate(pip_uotF(city='Torino', start_timestamp=start, end_timestamp=end)))

to_b_xF = np.arange(len(to_bF))
po_b_xF = np.arange(len(po_bF))
to_be_xF = np.arange(len(to_beF))

to_b_yF = [car['total'] for car in to_bF]
po_b_yF = [car['total'] for car in po_bF]
to_be_yF = [car['total'] for car in to_beF]

to_b_yF = np.asarray(to_b_yF)
po_b_yF = np.asarray(po_b_yF)
to_be_yF = np.asarray(to_be_yF)

# DA DECIDERE SE SERVE FILTRARE PARKINGS

# Parkings
to_pF = list(db['PermanentParkings'].aggregate(pip_uotF(city='Torino', start_timestamp=start, end_timestamp=end)))
po_pF = list(db['PermanentParkings'].aggregate(pip_uotF(city='Portland', start_timestamp=start, end_timestamp=end)))
to_peF = list(db['enjoy_PermanentParkings'].aggregate(pip_uotF(city='Torino', start_timestamp=start, end_timestamp=end)))

to_p_xF = np.arange(len(to_pF))
po_p_xF = np.arange(len(po_pF))
to_pe_xF = np.arange(len(to_peF))

to_p_yF = [car['total'] for car in to_pF]
po_p_yF = [car['total'] for car in po_pF]
to_pe_yF = [car['total'] for car in to_peF]

to_p_yF = np.asarray(to_p_yF)
po_p_yF = np.asarray(po_p_yF)
to_pe_yF = np.asarray(to_pe_yF)

# PLOTS
# ==========================================
# PLOTS
# ==========================================
# Bookings
# No FILTER
#plt.figure()
labels = [
        '24',
        '12\nMon',
        '24',
        '12\nTue',
        '24',
        '12\nWed',
        '24',
        '12\nThu',
        '24',
        '12\nFri',
        '24',
        '12\nSat',
        '24',
        '12\nSun',
        '24'
    ]
#plt.subplot(311)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
axes = [ax1, ax2, ax3, ax4]
ax1.plot(to_b_x, to_b_y, linewidth=1, label='Torino C2G')
ax1.plot(po_b_x, po_b_y, linewidth=1, label='Portland C2G')
ax1.plot(to_be_x, to_be_y, linewidth=1, label='Torino enjoy')
ax1.set_ylabel("Car per hour")
#ax1.legend(loc='upper right')
ax1.set_title("Bookings")
ax1.set_xlim(0, 167)
ax1.set_xticks(np.arange(len(to_b_x)+1, step=12))
ax1.set_xticklabels(labels)

ax1.grid(linestyle='--', linewidth=.4, which="both")

# FILTER

#plt.subplot(312)
ax2.plot(to_b_xF, to_b_yF, linewidth=1, label='Torino C2G')
ax2.plot(po_b_xF, po_b_yF, linewidth=1, label='Portland C2G')
ax2.plot(to_be_xF, to_be_yF, linewidth=1, label='Torino enjoy')
ax2.set_ylabel("Car per hour")
#ax2.legend(loc='upper right')
ax2.set_title("Bookings (Filter)")
ax2.set_xlim(0, 167)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


ax2.set_xticks(np.arange(len(to_b_x)+1, step=12))
ax2.set_xticklabels(labels)

ax2.grid(linestyle='--', linewidth=.4, which="both")


# Parkings
# NO FILTER

#plt.subplot(313)
ax3.plot(to_p_x, to_p_y, linewidth=1, label='Torino C2G')
ax3.plot(po_p_x, po_p_y, linewidth=1, label='Portland C2G')
ax3.plot(to_pe_x, to_pe_y, linewidth=1, label='Torino enjoy')
ax3.set_ylabel("Car per hour")
#ax3.legend(loc='upper right')
ax3.set_title("Parkings")
ax3.set_xlim(0, 167)

ax3.set_xticks(np.arange(len(to_b_xF)+1, step=12))
ax3.set_xticklabels(labels)

ax3.grid(linestyle='--', linewidth=.4, which="both")
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=1)
#plt.savefig(fname="plots/Book_UOT_3.png")
#plt.close()

# FILTER

#plt.subplot(224)
ax4.plot(to_p_xF, to_p_yF, linewidth=1, label='Torino C2G')
ax4.plot(po_p_xF, po_p_yF, linewidth=1, label='Portland C2G')
ax4.plot(to_pe_xF, to_pe_yF, linewidth=1, label='Torino enjoy')
ax4.set_ylabel("Car per hour")
#ax4.legend(loc='upper left')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax4.set_title("Parkings (Filter)")
ax4.set_xlim(0, 167)


ax4.set_xticks(np.arange(len(to_b_xF)+1, step=12))
ax4.set_xticklabels(labels)

ax4.grid(linestyle='--', linewidth=.4, which="both")
plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9, hspace=1)

plt.savefig(fname="../plots/Book_UOT_t.png")
plt.close()


