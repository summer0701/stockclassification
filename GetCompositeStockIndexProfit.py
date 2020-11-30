import urllib.request
import xml.etree.ElementTree as ET


class GetCompositeStockIndexProfit:
    def __init__(self):
        print("hello")
    def getXml(self):
        year_price = {}

        #with urllib.request.urlopen('https://fchart.stock.naver.com/sise.nhn?symbol=KOSDAQ&timeframe=month&count=500&requestType=0') as response:
        with urllib.request.urlopen('https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=month&count=500&requestType=0') as response:
            html = response.read().decode(response.headers.get_content_charset())
        root = ET.fromstring(html)

        for subChild in root:
            for child in subChild:
                data_list = child.attrib['data'].split('|')
                if int(data_list[0][0:4]) in range(2004,2020):
                    if data_list[0][4:6] in ('02','03'):
                        year_price[int(data_list[0])] =data_list[4]

        oldyear =0
        oldprice = 0
        for yp in year_price:
            if oldyear == 0:
                oldyear=yp
                oldprice = year_price[yp]
                continue
            if str(yp)[0:4] == str(oldyear)[0:4]:
                oldyear = yp
                oldprice = year_price[yp]
                continue
            else:
                print("year : " + str(oldyear) + "~" + str(yp) + " profit : " + str(float(year_price[yp])/float(oldprice)*100))
                oldyear = yp
                oldprice = year_price[yp]











gcsip = GetCompositeStockIndexProfit()
gcsip.getXml()
