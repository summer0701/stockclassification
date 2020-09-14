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
NUMBER_OF_FEATURE = 66
THRASH_HOLD = 110

class StockClassfication:
	size_S = 30
	def __init__(self):
		print ("stock classfication init:")

	def priceThrashhold(self):
		self.training_result_np = np.load('./training_result_np_save.npy')
		self.training_result_thrashhold = np.where(self.training_result_np >THRASH_HOLD,1,0)

		np.save('training_result_thrashold', self.training_result_thrashhold)

	def prepareData(self):

		with open('change_list.json', 'r', encoding='cp949', errors='ignore') as f:
			self.org_data = json.load(f)
		with open('change_price.json', 'r', encoding='cp949', errors='ignore') as f:
			self.org_result = json.load(f)
		print("tipping")
		data = np.empty([0,0] ,dtype=int)
		#self.training_data = np.empty([0,0] ,dtype=int)
		i = 0
		self.training_data = []
		self.training_result = []
		for od in self.org_data:
			for o in self.org_data[od]:
				data = np.append(data,o[1])


			try:
				self.training_result.append(self.org_result[od])
				self.training_data.append(data)

			except:
				print("error..."+od)
			data = []
			#print("generating..."+od)

		self.training_data_np = np.array(self.training_data)
		np.save('training_data_np_save',self.training_data_np)
		self.training_result_np = np.array(self.training_result)
		np.save('training_result_np_save', self.training_result)
		print("pop")

	def LLS0_VarianceMaximizer(self):
		self.LS0_Clustering()
		self.LS1_compressing()

	def LLS1_ImportanceMaximizer(self):
		print("LLS1_ImportanceMaximizer")

	def LLS2_c2weightProductor(self):
		print("LLS2_c2weightProductor")
	def LS0_Clustering_chart(self, cs):
		x = []
		y = []
		for i in range(self.size_S):
			x.append(i)
			y.append(np.count_nonzero(cs==i))

		print("cluster number ")
		print(x)
		print("cluster count ")
		print(y)
		plt.bar(x, y)
		plt.xlabel("cluster number")
		plt.ylabel("count")
		plt.show()




	def LS0_Clustering(self):
		print("LS0_Clustering")

		self.training_data_np = np.load('./training_data_np_save.npy')
		# 각 fisets끼리 클러스터를 해야하기때문에 행과 열을 바꾼다.
		self.tdnTrans = np.transpose(self.training_data_np)
		kmeans = KMeans(n_clusters=self.size_S, random_state=0,n_init=1000).fit(self.tdnTrans)

		print("fisets data")
		print("Shape" +str(self.tdnTrans.shape[0])+" " + str(self.tdnTrans.shape[1]))

		print(self.tdnTrans)
		self.result_cluster_s =kmeans.labels_
		self.LS0_Clustering_chart(self.result_cluster_s)
		np.save('result_cluster_s', self.result_cluster_s)
	def LS1_compressing_chart(self):
		print("LS1_compressing_chart")
		self.training_data_with_c2fisets = pickle.load(open('training_data_with_c2fisets2.txt', 'rb'))
		x =[]
		y =[]

		for tdwc2 in self.training_data_with_c2fisets:
			x.append(tdwc2["cl"])
			y.append(tdwc2["cw"])

		#plt.hlines(1,np.amin(x),0.5)  # Draw a horizontal line
		#plt.eventplot(y, orientation='horizontal', colors='b')
		#plt.axis('off')
		#plt.show()
		plt.plot(x,y, 'g+')
		plt.show()





	def LS1_compressing(self):
		self.sum = 0
		print("LS1_compressing")
		self.training_data_np = np.load('./training_data_np_save.npy')
		self.result_cluster_s = np.load('./result_cluster_s.npy')
		self.training_result_np_save = np.load('./training_result_np_save.npy')
		self.training_result_thrashold = np.load('./training_result_thrashold.npy')

		self.tdnTrans = np.transpose(self.training_data_np)
		self.training_data_pca = []
		self.clusterLabel = []
		self.numberOfFisets = self.training_data_np.shape[1]
		self.training_data_with_c2fisets = []
		pca = PCA(n_components=1)
		for i in range(0,self.size_S-1):
			result = np.where(self.result_cluster_s == i)
			if result[0].size == 0:
				continue
			cw = self.LS0_Cluster_Weighter(self.training_data_np.shape[1], result[0].size)
			self.kbinsPCA = self.KbinPCA(self.tdnTrans[result])


			print("PCA data" )
			print(self.kbinsPCA)
			print("Shape" +str(self.kbinsPCA.shape[0])+" " + str(self.kbinsPCA.shape[1]))
			cl = 0
			for r in self.tdnTrans[result]:
				cl = cl+self.LS1_CorrealtionWeighter_kbin(self.training_result_np_save, r)
			self.training_data_with_c2fisets.append({'cl':cl, 'cw':cw, 'data':self.kbinsPCA})
			self.LS1_compressing_chart()
		with open('training_data_with_c2fisets.txt', 'wb') as fp:
			pickle.dump(self.training_data_with_c2fisets, fp, protocol=pickle.HIGHEST_PROTOCOL)


		print("LS1_compressing fin")
	def KbinPCA(self, data):
		kdata=[]
		if data.shape[0] == 1:
			est = preprocessing.KBinsDiscretizer(n_bins=[10], encode='ordinal').fit(data.reshape(-1, 1))
			kdata.append(est.fit_transform(data.reshape(-1, 1)))
			npa = np.array(kdata)


			npa = npa.reshape(npa.shape[0], npa.shape[1])
			pca = PCA(n_components=1)
			r_pca = pca.fit_transform(np.transpose(npa))
			return r_pca
		for d in data:
			est = preprocessing.KBinsDiscretizer(n_bins=[10], encode='ordinal').fit(d.reshape(-1, 1))
			kdata.append(est.fit_transform(d.reshape(-1, 1)))

		npa = np.array(kdata)
		print("kdata")
		print(np.transpose(npa))
		npa = npa.reshape(npa.shape[0], npa.shape[1])
		pca = PCA(n_components=2)
		r_pca = pca.fit_transform(np.transpose(npa))
		return r_pca




	def LS1_CorrealtionWeighter_kbin(self, trt, tdp):

		est = preprocessing.KBinsDiscretizer(n_bins=[10], encode='ordinal').fit(tdp.reshape(-1, 1))
		tdp_bf = est.fit_transform(tdp.reshape(-1, 1))

		est_f = preprocessing.KBinsDiscretizer(n_bins=[10], encode='ordinal').fit(trt.reshape(-1, 1))
		trt_bf =  est_f.fit_transform(trt.reshape(-1, 1))
		result = np.corrcoef(trt_bf.reshape(1, 3203), tdp_bf.reshape(1, 3203))
		print("LS1_CorrealtionWeighter_bin")
		if np.isnan(result[0][1]):
			return 0
		self.sum = self.sum + np.absolute(result[0][1])
		print("=========trt==============")
		print(np.transpose(trt_bf))
		print("=========tdp===========")
		print(np.transpose(tdp_bf))
		return np.absolute(result[0][1])

	def LS1_CorrealtionWeighter_bin(self, trt, tdp):

			tdp_bf = preprocessing.Binarizer().fit(tdp.reshape(-1, 1))

			tdp_bin = tdp_bf.transform(tdp.reshape(-1, 1))
			#		trt_bf = preprocessing.Binarizer().fit(trt.reshape(-1, 1))
			#		trt_normalized = trt_bf.transform(trt.reshape(-1, 1))
			result = np.corrcoef(trt.reshape(1, 3203), tdp_bin.reshape(1, 3203))
			if np.isnan(result[0][1]):
				return 0

			print("LS1_CorrealtionWeighter_bin")
			self.sum = self.sum + np.absolute(result[0][1])
			return result[0][1]

	def LS1_CorrealtionWeighter_norm(self, trt, tdp):
		tdp_bf = preprocessing.Binarizer().fit(tdp.reshape(-1, 1))
		norm = preprocessing.normalize(tdp.reshape(-1, 1), norm='l2')
		tdp_normalized = tdp_bf.transform(tdp.reshape(-1, 1))
#		trt_bf = preprocessing.Binarizer().fit(trt.reshape(-1, 1))
#		trt_normalized = trt_bf.transform(trt.reshape(-1, 1))
		result  = np.corrcoef(trt.reshape(1,3203), norm.reshape(1,3203))
		if np.isnan(result[0][1]):
			return 0
		print("LS1_CorrealtionWeighter_norm")
		self.sum = self.sum + np.absolute(result[0][1])
		return result[0][1]

	def LS0_Cluster_Weighter(self,totalFisets, subfisets):
		print("LS0_Cluster_Weighter")
		return subfisets/totalFisets

	def lSVC_Wrap(self,X_train, X_test, y_train, y_test,y_testProfit):
		clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-10))
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
		print("Precision:", metrics.precision_score(y_test, y_pred))
		print("Recall:", metrics.recall_score(y_test, y_pred))
		print("Profit is :",np.sum(y_testProfit*y_pred)/np.sum(y_pred))

		print("end of test")
	def SVC_Wrap(self,X_train, X_test, y_train, y_test,y_testProfit):
		clf = svm.SVC(kernel='linear')
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
		print("Precision:", metrics.precision_score(y_test, y_pred))
		print("Recall:", metrics.recall_score(y_test, y_pred))
		print("Profit is :",np.sum(y_testProfit*y_pred)/np.sum(y_pred))

		y_result = y_testProfit[np.where(y_pred != 0)]
		for y in y_result:
			print(y)


		print("end of test")
	def boost_Wrap(self,X_train, X_test, y_train, y_test,y_testProfit):
		clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
		print("Precision:", metrics.precision_score(y_test, y_pred))
		print("Recall:", metrics.recall_score(y_test, y_pred))
		print("Profit is :", np.sum(y_testProfit * y_pred) / np.sum(y_pred))
		print("end of test")
	def mlp_Wrap(self,X_train, X_test, y_train, y_test,y_testProfit):
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (100, 100), random_state = 1)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
		print("Precision:", metrics.precision_score(y_test, y_pred))
		print("Recall:", metrics.recall_score(y_test, y_pred))
		print("Profit is :", np.sum(y_testProfit * y_pred) / np.sum(y_pred))
		print("end of test")

	def StockClassification(self):
		self.training_result_thrashold = np.load('./training_result_thrashold.npy')

		training_data =[]
		self.training_data_with_c2fisets = pickle.load(open('training_data_with_c2fisets2.txt', 'rb'))
		self.training_result_np_save = np.load('./training_result_np_save.npy')
		for tdwc in self.training_data_with_c2fisets:
			for i in range(0,int(tdwc['w'])+1):
				for td in np.transpose(tdwc['data']):
					training_data.append(td)


		training_np = np.transpose(np.array(training_data))
		result_np = np.array(self.training_result_thrashold).reshape(-1, 1)

		tp = np.array_split(training_np, 3)
		rnp = np.array_split(result_np, 3)
		rnpp = np.array_split(self.training_result_np_save, 3)


		X_train = np.append(tp[0], tp[1], axis=0)
		y_train = np.append(rnp[0], rnp[1], axis=0)
		X_test = np.array(tp[2])
		y_test = np.array(rnp[2])
		y_testProfit = np.array(rnpp[2])
		#X_train, X_test, y_train, y_test = train_test_split(training_np, result_np, test_size=0.2, random_state=109)  # 70% training and 30% test

		print("SVC_Wrap")
		self.SVC_Wrap(X_train, X_test, y_train, y_test,y_testProfit)
		print("lSVC_Wrap")
		self.lSVC_Wrap(X_train, X_test, y_train, y_test,y_testProfit)
		print("boost_Wrap")
		self.boost_Wrap(X_train, X_test, y_train, y_test,y_testProfit)
		print("mlp_Wrap")
		self.mlp_Wrap(X_train, X_test, y_train, y_test,y_testProfit)

	def LS2_WeightingProductor(self):
		print("LS2_WeightingProductor")

		with open('training_data_with_c2fisets.txt', 'rb') as fp:
			self.training_data_with_c2fisets = pickle.load(fp)
		totalC2 = 0
		kbins = []
		for tdwc in self.training_data_with_c2fisets:
			tdwc['c2'] = tdwc['cl']*tdwc['cw']
			totalC2 = totalC2+tdwc['c2']
			kbins.append(tdwc['c2'])
		kbins_trans = preprocessing.KBinsDiscretizer(n_bins=[2], encode='ordinal')
		ft = kbins_trans.fit_transform(np.array(kbins).reshape(-1, 1))


		# c2 를 전체 c2에 나눈다.
		checksum =0
		i=0
		for tdwc in self.training_data_with_c2fisets:
			tdwc['c2'] = tdwc['c2']/totalC2 * 100
			tdwc['w'] = ft[i][0]
			i=i+1

			checksum = checksum+tdwc['c2']

		with open('training_data_with_c2fisets2.txt', 'wb') as fp:
			pickle.dump(self.training_data_with_c2fisets, fp, protocol=pickle.HIGHEST_PROTOCOL)


class StockClassficationV2:
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
sc.kbinData()

#sc.prepareData()
#sc.LLS0_VarianceMaximizer()
#sc.priceThrashhold()
#sc.LS0_Clustering()
#sc.LS1_compressing()
#sc.LS1_compressing_chart()
#sc.LS2_WeightingProductor()
#sc.StockClassification()





