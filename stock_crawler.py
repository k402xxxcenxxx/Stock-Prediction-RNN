from urllib.request import urlopen
from urllib.error import HTTPError

import pandas as pd
import json
import time

class StockCrawler(object):

    def __init__(self):
        self.api_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=%s&stockNo=%s&_=%s"

    def fetchDailyByMonth(self, stock_code, start_date):
        ''' Fetch daily trading data per month. 
            stock_code: string of stock id
            start_date: string of date, e.g. 20200501

            return json dict
        '''
        timestamp = time.time()

        #　query data
        query_url = self.api_url % (start_date, stock_code, timestamp)

        try:
            result = urlopen(query_url)
        except HTTPError as e:
            print(e.code)
            print(e.read())  
            return False

        result = json.loads(result.read())

        if(result['stat'] != 'OK'):
            raise Exception("query fail, query_url=%s"%query_url)

        df = pd.DataFrame(result['data'])
        df.columns = result['fields']

        return(df)

    def fetchOldDataFromGithub(self, stock_code):
        github_url = "https://raw.githubusercontent.com/Asoul/tsec/master/data/%s.csv"

        #　query data
        query_url = github_url % stock_code

        try:
            result = urlopen(query_url)
        except HTTPError as e:
            print(e.code)
            print(e.read())  
            return u"".encode('utf-8')


        result = result.read()

        return result
            
    