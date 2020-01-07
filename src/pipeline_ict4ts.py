


def pip_hmt(start_timestamp, end_timestamp, city='Torino'):
    """query for retrieving how many bookings/parkings were doing in a specific time"""

    pipeline = [
    {"$match":
         {"city": city}
    #{"init_time": {"$gte": start_timestamp}},
    #{"final_time": {"$lte": end_timestamp}}
    },
    {"$group": {
        "_id": {"day": {"$dayOfYear": "$init_date"}, "year": {"$year": "$init_date"}},
        "plate": {"$addToSet": "$plate"}},

    },
     {
         "$unwind": "$plate"
     },
    {
        "$group":  {
            "_id": {"day": "$day", "year": "$year"},
            "Count": {"$sum": 1}
        }
    }

    ]

    return pipeline


def pip_d(start_timestamp, end_timestamp, city='Torino'):
    """pipeline for querying how much time bookings/Parkings take: duration contains the booking/parking time for each
    car"""

    pipeline = [
        {"$match":
             {"city":  city,
              'init_time': {'$gte': start_timestamp},
              'final_time': {'$lte': end_timestamp}
              }
         },

        {"$project": {
            "_id": 1,
            "init_date": 1,
            "day": {'$dayOfMonth': '$init_date'},
            "duration": {"$subtract": ['$final_time', '$init_time']},
            "fuel": {'$subtract': ['$final_fuel', '$init_fuel']}
            }
        },
        {
            '$match': {
                'duration': {'$gte': 180, "$lt": 5000},
                'fuel': {'$ne': 0}
            }
        },
        {
            '$group': {
                '_id': {'day': '$day'},
                'date': {"$first": "$init_date"},
                'StdDuration': {'$stdDevPop': '$duration'},
                'AvgDuration': {'$avg': '$duration'},
                'travel': {'$push': "$duration"}


            }
        },

        {
            "$sort":
                {"_id.day": 1, "date": 1}
        }


    ]

    return pipeline


def pip_uot(start_timestamp, end_timestamp, city='Torino', objective=True):
    """pipeline Utilization Over Time: total contains the total number of booking/parking for each hour"""


    if objective:
        pipeline = [
        {
            '$match': {
                'city': city,
                'init_time': {
                    '$gte': start_timestamp,
                    '$lte': end_timestamp
                }
            }
        }, {
            "$project":
                {'_id': 1,
                 "dd": {"$dayOfWeek": "$init_date"},
                 "hh": {"$hour": "$init_date"},
                 "init_date": 1,
                 }
        },
        {
            '$group': {
                '_id': {'day': '$dd', 'hour': '$hh'},
                'date': {"$first": "$init_date"},
                'total': {'$sum': 1},

            }
        },

        {
            '$sort': {'_id': 1}
        }
        ]
        return pipeline
    else:
        pipeline = [
            {
                '$match': {
                    'city': city,
                    'init_time': {
                        '$gte': start_timestamp,
                        '$lte': end_timestamp
                    }
                }
            }, {
                "$project":
                    {'_id': 1,
                     "dd": {"$dayOfYear": "$init_date"},
                     "year": {"$year": "$init_date"},
                     "init_date": 1,
                     }
            },
            {
                '$group': {
                    '_id': {'day': '$dd', 'year': '$year'},
                    'total': {'$sum': 1},

                }
            },

            {
                '$sort': {'_id': 1}
            }
        ]
        return pipeline





def pip_uotF(start_timestamp, end_timestamp, city='Torino', objective='week'):
    """pipeline Utilization Over Time with filter: total contains the total number of booking/parking for each hour,
    by taking into account only the cars that really were used."""


    if objective == 'week':

        pipeline = [
            {
            '$match': {
                'city': city,
                'init_time': {
                    '$gte': start_timestamp,
                    '$lte': end_timestamp
                }
            }
        }, {
            "$project":
                {'_id': 1,
                 "dd": {"$dayOfWeek": "$init_date"},
                 "hh": {"$hour": "$init_date"},
                 "init_date": 1,
                 "total_fuel": {'$subtract': ['$final_fuel', '$init_fuel']},
                 "duration": {"$subtract": ['$final_time', '$init_time']},
                }
        },
        {
            '$match': {'total_fuel': {'$ne': 0}, "duration": {"$gte": 300, "$lt": 5000}}
        },
        {
            '$group': {

                '_id': {'day': '$dd', 'hour': '$hh'},
                'date': {'$first': '$init_date'},
                'total': {'$sum': 1},
                }
        }
        ]
        return pipeline



    elif objective == 'year':

        pipeline = [
            {
                '$match': {
                    'city': city,
                    'init_time': {
                        '$gte': start_timestamp,
                        '$lte': end_timestamp
                    }
                }
            }, {
                "$project":
                    {'_id': 1,
                     "hh": {"$hour": "$init_date"},
                     "dd": {"$dayOfYear": "$init_date"},
                     "year": {"$year": "$init_date"},
                     "init_date": 1,
                     "total_fuel": {'$subtract': ['$final_fuel', '$init_fuel']},
                     "duration": {"$subtract": ['$final_time', '$init_time']},
                     }
            },
            {
                '$match': {'total_fuel': {'$ne': 0}, "duration": {"$gte": 150, "$lte": 5000}}
                #'$match': {"duration": {"$gte": 100, "$lte": 8000}}
            },
            {
                '$group': {

                    '_id': {'year': '$year', 'day': '$dd', 'hour': '$hh'},
                    'date': {'$first': '$init_date'},
                    'total': {'$sum': 1},
                }
            }

            #{
                #'$sort': {'_id': 1}
            #}
        ]
        return pipeline



