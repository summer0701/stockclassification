from sklearn import preprocessing
import numpy as np
import requests
import codecs
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from scipy import stats
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import dump, load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

NUMBER_OF_FEATURE = 66
THRASH_HOLD = 120
RISK_FREE_RETURN = 102



class StockClassficationV2:
	arrayList = ['0405','0506','0607','0708','0809'
						,'0910','1011','1112','1112',
						'1213','1314','1415','1516',
						'1617','1718']

	def __init__(self):
		print("__init__")
	def prepareData_chart(self):
		training_data = []
		with open('training_data_set.txt', 'rb') as fp:
			training_set = pickle.load(fp)


		for ts in training_set:
			training_data.append(training_set[ts]["d"])


		npa = np.array(training_data)

		plt.boxplot(np.transpose(npa)[35:70])
		plt.xlabel("fisets")
		plt.ylabel("value")
		plt.show()
	def kbinChart(self,data):

		for d in data:
			print(d)

	def kbinData(self):
		print("kbinData")
		training_data = []
		with open('training_data_set.txt', 'rb') as fp:
			training_set = pickle.load(fp)
		npa = np.empty([0,70] ,dtype=int)
		ts_del_list = []
		for ts in training_set:
			try:
				npa = np.append(npa, np.array(training_set[ts]["d"]), axis=0)
			except:
				ts_del_list.append(ts)
				continue
		for tdl in ts_del_list:
			del training_set[tdl]
		est = []

		for n in npa.transpose():
			est.append(preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal').fit_transform(n.reshape(-1,1)))

		np_est = np.array(est).transpose()[0]
		self.kbinChart(np_est)
		i=0
		for ts in training_set:
			training_set[ts]["kbinD"] = np_est[i]
			i+=1
		with open('training_data_set_kbin', 'wb') as fp:
			pickle.dump(training_set, fp, protocol=pickle.HIGHEST_PROTOCOL)
		print("done")
	def spliteData(self):


		with open('training_data_set.txt', 'rb') as fp:
			training_set = pickle.load(fp)
		for al in self.arrayList:
			training_data = np.empty([0, 70], dtype=int)
			training_result = np.empty([0, 1], dtype=int)
			for ts in training_set:
				if al in ts: # test data
					try:
						if len(training_set[ts]["d"][0]) == 70:
							training_data = np.append(training_data, np.array(training_set[ts]["d"]), axis=0)
							training_result = np.append(training_result, np.array(training_set[ts]["r"]).reshape(1,1), axis=0)
					except:
						print("data is missing continue")
			with open('training_data_'+al+'.txt', 'wb') as fp:
				pickle.dump(training_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
			with open('training_result_'+al+'.txt', 'wb') as fp:
				pickle.dump(training_result, fp, protocol=pickle.HIGHEST_PROTOCOL)




		npa = np.empty([0, 70], dtype=int)
		# 특정연도만 추려와서 스케일링 한다.

	def training_dnn(self):

		for a in self.arrayList:
			with open('training_data_'+a+'.txt', 'rb') as fp:
				training_data = pickle.load(fp)
			with open('training_result_'+a+'.txt', 'rb') as fp:
				training_result = pickle.load(fp)


			scaler = preprocessing.RobustScaler().fit(training_data)
			dump(scaler, 'scaler_'+a)

			scaled_training_data = scaler.transform(training_data)


			thrashhold_training_result = np.where(training_result >THRASH_HOLD,1,0)
			model = self.build_DNN(training_data.shape[1])
			model.fit(scaled_training_data, thrashhold_training_result, epochs=300, batch_size=10,verbose=0)
			model.save('./model_'+a)
			#print(model.evaluate(scaled_training_data,thrashhold_training_result))
			#print((model.predict(scaled_training_data) > 0.5).astype(int))

		
		print("pause")

	def build_fDNN(self,input):

		model = Sequential()
		model.add(Dense(512, activation='sigmoid', input_dim=input))
		model.add(Dropout(0.2))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(128, activation='sigmoid'))
		model.add(Dropout(0.2))
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(64, activation='sigmoid'))
		model.add(Dropout(0.2))
		model.add(Dense(1,activation='sigmoid'))

		#model.summary()

		model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		return model

	def build_DNN(self,input):

		model = Sequential()
		model.add(Dense(512, activation='sigmoid', input_dim=input))
		model.add(Dropout(0.2))
		model.add(Dense(128, activation='sigmoid'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(1,activation='sigmoid'))

		#model.summary()

		model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		return model
	def build_training_set(self):

		test_list = ['1415','1516','1617']
		for alist in self.arrayList:
			self.build_detail_training_set(alist)

	def final_training_set(self,data, scalers, models):
		# predict 해서 training set을 만든다.
		mid_training_data = np.empty((data.shape[0], 0))
		for alist in self.arrayList:
			transformed_data = scalers[alist].transform(data)
			mid_predicted_data = (models[alist].predict(transformed_data)>0.5).astype(int)
			mid_training_data = np.hstack((mid_training_data, mid_predicted_data))
		return mid_training_data
	def build_detail_training_set(self, year):
		training_data = np.empty([0, 70], dtype=int)
		training_result = np.empty([0, 1], dtype=int)
		test_data = np.empty([0, 70], dtype=int)
		test_result = np.empty([0, 1], dtype=int)
		model = self.build_DNN(training_data.shape[1])
		scalers = {}
		models = {}
		# 해당년도를 제외하고 트레이닝 세트를 만든다.
		for alist in self.arrayList:
			if alist != year:
				with open('training_data_'+alist+'.txt', 'rb') as fp:
					td = pickle.load(fp)
				with open('training_result_'+alist+'.txt', 'rb') as fp:
					tr = pickle.load(fp)
				training_data = np.append(training_data, td, axis=0)
				training_result = np.append(training_result, tr, axis=0)
				#  scaler 로드
				scalers[alist] = load('scaler_' + str(alist))
				models[alist] = keras.models.load_model('./model_'+str(alist))
				

			else:
				with open('training_data_'+alist+'.txt', 'rb') as fp:
					td = pickle.load(fp)
				with open('training_result_'+alist+'.txt', 'rb') as fp:
					tr = pickle.load(fp)
				test_data = np.append(test_data, td, axis=0)
				test_result = np.append(test_result, tr, axis=0)
				scalers[alist] = load('scaler_' + str(alist))
				models[alist] = keras.models.load_model('./model_'+str(alist))

		final_training_data = self.final_training_set(training_data,scalers,models)
		fmodel = self.build_fDNN(final_training_data.shape[1])
		thrashhold_training_result = np.where(training_result > THRASH_HOLD, 1, 0)
		fmodel.fit(final_training_data, thrashhold_training_result, epochs=3000, batch_size=150,verbose=0)




		final_test_data = self.final_training_set(test_data, scalers, models)
		thrashhold_test_result = np.where(test_result > THRASH_HOLD, 1, 0)

		final_test_result = (fmodel.predict(final_test_data) > 0.5).astype(int)
		print(str(year) + "year accrucy : " + str(fmodel.evaluate(final_test_data,thrashhold_test_result)[1]))
		t = final_test_result * test_result
		true_data = t[np.where(t != 0)[0]]
		tp = np.sum(true_data > RISK_FREE_RETURN)/np.sum(final_test_result)
		fp = np.sum(true_data < RISK_FREE_RETURN)/np.sum(final_test_result)
		print("true positive:", tp)
		print("false positive:", fp)
		print("avg profit of "+str(year) + " is " + str(np.sum(final_test_result * test_result)/np.sum(final_test_result)))
		print("profit of " + str(year) + " is ")
		print("selection count is " + str(np.sum(final_test_result)))
		print((final_test_result * test_result).transpose())







		pass



	def prepareData(self):

		with open('change_list.json', 'r', encoding='cp949', errors='ignore') as f:
			org_data = json.load(f)
		with open('change_price.json', 'r', encoding='cp949', errors='ignore') as f:
			org_result = json.load(f)

		data = np.empty([0,0] ,dtype=int)
		#self.training_data = np.empty([0,0] ,dtype=int)
		i = 0
		training_set={}
		training_data = []
		training_result = []
		for od in org_data:
			for o in org_data[od]:
				data = np.append(data,o[1])


			try:
				if len(data) != 70:
					print(data)
				training_result.append(org_result[od])
				training_data.append(data)

			except:
				print("error..."+od)
				data = []
				training_data = []
				training_result = []
				continue
			data = []
			training_data = []
			training_result = []
			training_set[od] = {"r" : training_result,"d": training_data}

			#print("generating..."+od)

		with open('training_data_set.txt', 'wb') as fp:
			pickle.dump(training_set, fp, protocol=pickle.HIGHEST_PROTOCOL)





sc = StockClassficationV2()
#sc.prepareData()
#sc.prepareData_chart()
#sc.kbinData()
#sc.spliteData()
sc.training_dnn()
sc.build_training_set()


#sc.prepareData()
#sc.LLS0_VarianceMaximizer()
#sc.priceThrashhold()
#sc.LS0_Clustering()
#sc.LS1_compressing()
#sc.LS1_compressing_chart()
#sc.LS2_WeightingProductor()
#sc.StockClassification()





