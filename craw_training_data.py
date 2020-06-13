from stock_crawler import StockCrawler
import pandas as pd
from dateutil import rrule
from datetime import datetime

from os import mkdir
from os.path import isdir, isfile

class CrawlStockMan():
    def __init__ (self, start_date_str, end_date_str, stock_code_list):
        self.prefix = "data"
        if not isdir(self.prefix):
            mkdir(self.prefix)

        self.dateFormatter = "%Y%m%d"
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.stock_code_list = stock_code_list

    def set_time_range(self):
        self.start_date = datetime.strptime(self.start_date_str, self.dateFormatter)
        self.end_date = datetime.strptime(self.end_date_str, self.dateFormatter)

    def crawl(self):
        sc = StockCrawler()

        for index in range(len(self.stock_code_list.index)):

            stock_code = self.stock_code_list['code'][index]
            if self.checkDataExist(stock_code):
                continue

            print("Start to craw %s" % self.stock_code_list['name'][index])

            data = sc.fetchOldDataFromGithub(stock_code)
            self.saveOldData(stock_code, data)

            # for dt in rrule.rrule(rrule.MONTHLY, dtstart=self.start_date, until=self.end_date):

            #     date_str = dt.strftime(self.dateFormatter)
            #     data = sc.fetchDailyByMonth(stock_code, date_str)

            #     print(data)

    def saveOldData(self, stock_id, data):
        f = open('{}/{}.csv'.format(self.prefix, stock_id), 'wb')
        f.write(data)
        f.close()

    def checkDataExist(self, stock_id):
        return isfile('{}/{}.csv'.format(self.prefix, stock_id));
               
stock_code_list = pd.read_csv('stock_code.csv')
crawlMan = CrawlStockMan("20100101", "20100101", stock_code_list)
crawlMan.set_time_range()
crawlMan.crawl()