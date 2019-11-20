import requests
import json
import xlrd
import os


PER_THRASHHOLD = 6
EPS_GROWTH = 15
class ConsensusFinder:
	def __init__(self):
		print('start point')
	def get_code(self):
		workbook_read = xlrd.open_workbook('code_data.xls')
		sheet_list = workbook_read.sheets()
		sl = sheet_list[0]
		codes = {}
		for i in range(1, len(sl.col(0))):
			try:
				codes[sl.col(1)[i].value] = sl.col(2)[i].value
				print(sl.col(1)[i])
			except:
				continue
		with open('cfsc.json', 'w', encoding='utf-8') as make_file:
			json.dump(codes, make_file, indent="\t")

	def f2(self,x):
		return x[1]['SALES']
	def order_peg(self):
		with open('est_list.json', 'r') as f:
			est_list = json.load(f)
			res = sorted(est_list.items(), key=self.f2)
			for r in res:
				print(str(r[0]) + ' sales : '+ str(r[1]['SALES']) +' ' + r[1]['name'] + ": PEG : " + str(r[1]['PEG']) + ' EPS_GROWTH : ' + str(r[1]['EPS_GROWTH']) + ' E_PER : ' + str(r[1]['E_PER']) )

	def get_consensus(self):
		with open('cfsc.json', 'r') as f:
			cfsc = json.load(f)
		est_list = {}
		for cf in cfsc:
			comp = cf
			url = 'https://navercomp.wisereport.co.kr/company/ajax/c1050001_data.aspx?flag=2&cmp_cd='+comp+'&finGubun=MAIN&frq=0&chartType=svg'

			try:
				r = requests.get(url)
			except:
				print('error')
			con = json.loads(r.text)['JsonData']

			if len(con) < 7:
				print('최신 컨센이 없네요.')
				continue
			try:
				eps4 = int(con[4]['EPS'].replace(',',''))
				eps5 = int(con[5]['EPS'].replace(',',''))
				eps6 = int(con[6]['EPS'].replace(',',''))
				sales = int(float(con[6]['SALES'].replace(',', '')))

				eper = float(con[6]['PER'])
				eps_growth =((eps6 - eps4)/eps4) * 50
				if eps_growth < 0:
					continue
				if eper < 0:
					continue
				if eper > PER_THRASHHOLD:
					continue
				if eps_growth < EPS_GROWTH:
					continue
				peg = eper/eps_growth

				est_list[cf] = {}
				est_list[cf]['name'] = cfsc[cf]
				est_list[cf]['PEG'] = peg
				est_list[cf]['EPS_GROWTH'] = eps_growth
				est_list[cf]['E_PER'] = eper
				est_list[cf]['SALES'] = sales


			except:
				print('최신 컨센이 없네요.')
				continue
		with open('est_list.json', 'w', encoding='utf-8') as make_file:
			json.dump(est_list, make_file, indent="\t")

cf = ConsensusFinder()
#cf.get_code()
#cf.get_consensus()
cf.order_peg()