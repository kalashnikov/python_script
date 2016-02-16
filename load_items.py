# -*- coding: utf-8 -*-

import json
# from pprint import pprint

with open('items.json') as data_file:
        data = json.load(data_file)

for items in data:
    if "id=" not in items['u']:
        continue
    print "\"" + items['w'] + "\",\"" + items['a'] + "\"," + items['u']
