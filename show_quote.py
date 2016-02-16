# -*- coding: utf-8 -*-

from pymongo import MongoClient

uri = "mongodb://localhost:27017"
client = MongoClient(uri)

db = client['quote']
collect = db['zh']

for post in collect.find():
    print post['w'] + " - " + post['a']
