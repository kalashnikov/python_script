import json
import quandl 
import pandas as pd
import requests as r
import os
import sys

# Load data using Gurufocus API 
# Process data into DataFrame object and save as csv format
def get_financials_data_by_token(name):
    data_json = r.get('https://www.gurufocus.com/api/public/user/{}/stock/{}/financials'.format(os.environ['GURU_TOKEN'], name), headers={'User-Agent': 'Mozilla/5.0'}).json()

    for t in data_json['financials'].keys():
        df_list = []

        # Accmulate data into DataFrame list 
        keys = list(data_json['financials'][t].keys())
        keys.remove('Fiscal Year') # Used as Index 
        for k in keys:
            df_list.append(pd.DataFrame(data=data_json['financials'][t][k], index=data_json['financials'][t]['Fiscal Year']))

        # Concate lists
        df = pd.concat(df_list, axis=1)
        df.to_csv('{}_{}.csv'.format(name, t))


# Load data using Gurufocus API 
# Process data into DataFrame object and save as csv format
def get_price_by_token(name):
    quandl.ApiConfig.api_key = os.environ['QUANDL_TOKEN']
    data = quandl.get("WIKI/{}".format(token))
    data['Weekday']=data.index.weekday
    data['Dif']= data['Open'] - data['Close']
    data['Range']= data['High'] - data['Low']
    data.to_excel('{}.xlsx'.format(name), "Price")
    #data.to_csv('{}_price.csv'.format(name))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Please provide token.")
        exit()

    token = sys.argv[1]
    get_financials_data_by_token(token)
    get_price_by_token(token)
    print('### Finish load {} financials data and output to csv files.'.format(token))
