# ICT4TS_Git
# First ICT4TS laboratory: 
**Lab goals**
The laboratory part of the ICT4TS focuses on the analysis of free floating car sharing data collected from open systems in the Internet. Data has been collected from websites of FFCS systems like car2go or Enjoy in Italy, and made available through a MondoDB database. The goal of the laboratory is twofold:
• to allow students to get used to ICT technologies typically used in the backend of smart society applications – database and remote server access, and writing analytics using simple scripts
• to allow students to work on real data, trying to extract useful information specific to the application domain – transport system in this case – and get used data pre-processing and filtering.
Accessing the data
Data is stored in a MongoDB server running on the server bidgatadb.polito.it:27017. It offers read only access to clients connected a network to the following database:
• Collection name: Carsharing
• User: ictts
• Password: Ictts16!
• Requires SSL with self signed certificates

**Collections**
The system exposes 4 collections for Car2Go, which are updated in real time. Those are
• "ActiveBookings": Contains cars that are currently booked and not available
• "ActiveParkings": Contains cars that are currently parked and available
• “PermanentBookings": Contains all booking periods recorded so far
• "PermanentParkings": Contains all parking periods recorded so far
The same collections are available for Enjoy as well. Names are self-explanatory:
• "enjoy_ActiveBookings": Contains cars that are currently booked and not available
• "enjoy_ActiveParkings": Contains cars that are currently parked and available
• ”enjoy_PermanentBookings": Contains all booking periods recorded so far
• "enjoy_PermanentParkings": Contains all parking periods recorded so far
For Torino and Milano, the system augments the booking information with additional information obtained from Google Map service: walking, traveling, and public transportation alternative possibilities. Not all of them are available, due to the limited number of queries google allows.
