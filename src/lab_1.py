from connection import Connection
from datetime import datetime


client = Connection()
db = client.db


interest_collections = ['ActiveBookings', 'ActiveParkings', 'PermanentBookings',
                        'PermanentParkings', 'enjoy_ActiveBookings',
                        'enjoy_ActiveParkings']
# how many documents are present in each collections.
numberDocuments_dict = {}
for collection in interest_collections:
    numberDocuments_dict[collection] = db[collection].estimated_document_count()

# What is the start time and end time for each collection.
for collection in interest_collections:
    if collection not in ['enjoy_ActiveParkings', 'enjoy_ActiveBookings']:
        cursor_start = db[collection].find().sort([('init_time', 1)]).limit(1)
        cursor_end = db[collection].find().sort([('init_time', -1)]).limit(1)
        dict_start = cursor_start.next()
        dict_end = cursor_end.next()
        time_start = datetime.fromtimestamp(dict_start['init_time']).isoformat(sep='|')
        time_end = datetime.fromtimestamp(dict_end['init_time']).isoformat(sep='|')
        print(f"Collection: {collection} \n time start: {time_start} \n time end: {time_end}")

print("\n")
# how many cars are available for each cities
dict_cars = {'Torino': [], 'Portland': []}
for city in dict_cars.keys():
    for collection in interest_collections:
        value = db[collection].count_documents({'city': city})
        dict_cars[city].append({'Collection': collection, 'Value': value})









