import os
import xlrd
import json
import numpy as np
import pandas_datareader
import datetime
import requests
import xmltodict
import operator
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
price_standard = 110
class ExcelOpen:

    def __init__(self):
        print('starting point')

    def prepare_training_data(self,clf):
        with open('change_list.json', 'r') as f:
            cl = json.load(f)
        with open('change_price.json', 'r') as f:
            cp = json.load(f)

        training_data = np.empty((0,0), int)
        training_datas = np.empty((0,70), int)
        training_result =  np.empty((0,0), int)
        cp_x = sorted(cp.items(), key=operator.itemgetter(1))
        index = 0
        for x in cp_x:
            if x[1] > price_standard:
                training_result = np.append(training_result, [1])
            else:
                training_result = np.append(training_result, [0])
            for c in cl[x[0]]:
                training_data = np.append(training_data, [c[1]])

            training_datas = np.append(training_datas, training_data.reshape(1,-1),axis=0)
            training_data =  np.empty((0,0), int)
            index = index + 1
        print("TRUE(10% 이상인값) " + str(np.count_nonzero(training_result == 1)))
        print("FALSE(10% 이하인값) " + str(np.count_nonzero(training_result == 0)))
        print("end of data")
        score = 0

        total = 0
        profit = {}
        profit_count = {}
        #clf = svm.LinearSVC(random_state=0, tol=1e-5)
        if clf == 'svm':
            clf = svm.SVC(kernel='linear', class_weight={1:6, 0:4})
        elif clf == 'ada':
            clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        else:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)

        scaler = StandardScaler()
        scaler.fit(training_datas)
        training_datas = scaler.transform(training_datas)
        for idx in range(0,training_datas.shape[0] ) :
            #print("종목 및 연도: " + cp_x[idx][0] + " 수익률 :" + str(cp_x[idx][1]))

            testdata = training_datas[idx]
            testrresult = training_result[idx]
            if idx ==0:
                rtraining_data = training_datas[idx + 1:training_datas.shape[0]]
                rtraining_result = training_result[idx + 1:training_result.shape[0]]
            elif idx == training_datas.shape[0]:
                rtraining_data = training_datas[0:idx]
                rtraining_result = training_result[0:idx]
            else:
                rtraining_data = np.vstack((training_datas[0:idx], training_datas[idx + 1:training_datas.shape[0]]))
                rtraining_result = np.hstack((training_result[0:idx], training_result[idx + 1:training_result.shape[0]]))

            clf.fit(rtraining_data, rtraining_result)
            r = clf.predict(testdata.reshape(1, -1))
            if int(r) == 1:
                year = int(cp_x[idx][0].split('_')[1][2:4]) + 1
                try:
                    profit[year]
                except:
                    profit[year]=0
                    profit_count[year]=0

                profit[year] = profit[year] + int(cp_x[idx][1])
                profit_count[year] = profit_count[year] + 1
             # print('test value : ' + str(r) + 'real value : ' + str(testrresult))
            if r == testrresult:
                score = score + 1

            total = total+1
        # print ("total score = " + str(score/total))
        for year in profit:
            p = profit[year]/profit_count[year]
            print('year : ' + str(year)  + 'profit : ' + str(p) + 'count : ' + str(len(profit)))








    def getprice(self,workbook_read_name,year):
        url =  'https://fchart.stock.naver.com/sise.nhn?symbol='+ workbook_read_name + '&timeframe=month&count=500&requestType=0'
        sprice = 0
        eprice = 0
        try:
            r = requests.get(url)
        except:
            print('error')
        doc = xmltodict.parse(r.text)
        for i in range(0,len(doc['protocol']['chartdata']['item'])):
          if doc['protocol']['chartdata']['item'][i]['@data'].split('|')[0][0:6] == year + '03':
              if sprice > 0:
                  continue
              sprice = doc['protocol']['chartdata']['item'][i]['@data'].split('|')[4]
          if doc['protocol']['chartdata']['item'][i]['@data'].split('|')[0][0:6] == year + '12':
              if eprice > 0:
                  continue
              eprice = doc['protocol']['chartdata']['item'][i]['@data'].split('|')[4]
              break
        rprice = int(int(eprice)/int(sprice)*100)
        if rprice > 2000:
            print('a')
        return rprice
    def loadjson(self):
        with open('stock_list.json', 'r') as f:
          slist = json.load(f)
        change_list={}
        change_price = {}
        for sl in slist:
          for sy in slist[sl]['year']:
              try:
                  np1 = np.asarray(list(slist[sl]['year'][str(int(sy)+1)].values())) + 1
                  np2 = np.asarray(list(slist[sl]['year'][sy].values())) + 1
                  np3 = np.subtract(np1,np2)
                  np4 = np.divide(np3,np2)*100
                  np4_list = list(np.nan_to_num(np4))
                  np4_list = list(map(int, np4_list ))

                  change_key = slist[sl]['year'][str(int(sy) + 1)].keys()
                  zipobj = list(zip(change_key, np4_list))
                  change_list_name = slist[sl]['name'] + '_' + sy[2:4] + str(int(sy) + 1)[2:4]
                  change_list[change_list_name] = zipobj
                  rprice = self.getprice(sl, str(int(sy) + 1))
                  change_price [change_list_name] = rprice
              except:
                  print(sl + " " + sy)
        print('completed')
        with open('change_list.json', 'w', encoding='utf-8') as make_file:
          json.dump(change_list, make_file, indent="\t")
        with open('change_price.json', 'w', encoding='utf-8') as make_file:
          json.dump(change_price, make_file, indent="\t")


    def OpenWorkbook(self):
        cur_dir = os.getcwd()
        sands_folder = cur_dir + "/sands"
        file_list = os.listdir(sands_folder)
        stock_list = {}
        for fl in file_list:
            workbook_read_name = fl
            code = fl.split('.')[0]
            workbook_read = xlrd.open_workbook(os.path.join(sands_folder , workbook_read_name))
            sheet_list = workbook_read.sheets()


            # 사업보고서
            sl = sheet_list[0]
            stock_list[code] = {}
            stock_list[code]['name'] = sl.col(1)[1].value
            stock_list[code]['year'] = {}
            sl = sheet_list[1]
            for x in range(1,len(sl.row(0))):
              year = sl.row(0)[x].value.replace("년","")
              try: # 처음 입력 할경우에서는
                  stock_list[code]['year'][year]
                  continue # 기존에 입력되어있으면 정정공시이기때문에 건너뛴다.
              except:
                print(stock_list[code]['name'] + " retrieve cashflow..." )
                sl = sheet_list[1]
                stock_list[code]['year'][year] = {}
                for i in range (1,len(sl.col(0))):
                 #   if i in [5,15,16,17,18,19,20,21,22,23]:
                 #       continue
                    stock_list[code]['year'][year][sl.col(0)[i].value] = sl.col(x)[i].value
                print(stock_list[code]['name'] + " retrieve balancesheet...")
                sl = sheet_list[2]
                for i in range (1,len(sl.col(0))):
                    stock_list[code]['year'][year][sl.col(0)[i].value] = sl.col(x)[i].value
                sl = sheet_list[3]
                print(stock_list[code]['name'] + " retrieve income statement...")
                for i in range (1,len(sl.col(0))):
                    stock_list[code]['year'][year][sl.col(0)[i].value] = sl.col(x)[i].value
            sl = sheet_list[4]
            #stock_list[code]['price'] = {}
            #print(" retrieve price" )
            #for i in range(1,len(sl.col(0))):
            #  stock_list[code]['price'][sl.col(0)[i].value] = sl.col(1)[i].value
            print(stock_list[code]['name'] +  'completed')
            with open('stock_list.json', 'w', encoding='utf-8') as make_file:
                json.dump(stock_list, make_file, indent="\t")













eo = ExcelOpen()
#eo.OpenWorkbook()
#eo.loadjson()
eo.prepare_training_data('svm')
eo.prepare_training_data('ada')
eo.prepare_training_data('ne')