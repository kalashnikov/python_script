# -*- coding: utf-8 -*-

import csv
from pymongo import MongoClient

uri = "mongodb://localhost:27017"
client = MongoClient(uri)

db = client['quote']
collect = db['zh']

for post in collect.find():
    print post['w'], post['a']

with open('zh_quote.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        obj = {"w": row[0], "a": row[1]}
        if collect.find(obj).count() == 0:
            collect.insert_one(obj)

with open('out.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        obj = {"w": row[0], "a": row[1]}
        if collect.find(obj).count() == 0:
            collect.insert_one(obj)

print collect.count()
