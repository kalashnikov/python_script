#coding: utf8

from pymongo import MongoClient

uri = "mongodb://localhost:27017" 
client = MongoClient(uri)

db = client['quote']
collect = db['zh']

for post in collect.find():
    print post

import csv
with open('zh_quote.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        obj = { "w":row[0], "a":row[1] }  
        collect.insert(obj)
        #print obj 
