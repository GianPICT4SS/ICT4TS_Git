from pymongo import MongoClient


class Connection():

    def __init__(self, url_cluster="bigdatadb.polito.it", port=27017, authSource='carsharing',
                 username='ictts', psw='Ictts16!'):

        self.url_cluster = url_cluster
        self.port = port
        self.authSource = authSource
        self.username = username
        self.psw = psw
        self.client = MongoClient(host=self.url_cluster, port=self.port,
                     ssl=True, authSource=self.authSource,
                     tlsAllowINvalidCertificates=True)

        self.db = self.client[self.authSource]
        self.db.authenticate(self.username, self.psw)
