from sklearn import preprocessing
import numpy as np
import requests
import codecs
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
NUMBER_OF_FEATURE = 66
THRASH_HOLD = 103

class StockClassfication:
	def __init__(self):
		print ("stock classfication init:")

	def priceThrashhold(self):
		self.training_result_np = np.load('./training_result_np_save.npy')
		self.training_result_thrashhold = np.where(self.training_result_np >103,1,0)

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

	def LS0_Clustering(self):
		print("LS0_Clustering")
		self.size_S = 15
		self.training_data_np = np.load('./training_data_np_save.npy')
		# 각 fisets끼리 클러스터를 해야하기때문에 행과 열을 바꾼다.
		self.tdnTrans = np.transpose(self.training_data_np)
		kmeans = KMeans(n_clusters=self.size_S, random_state=0,n_init=100).fit(self.tdnTrans)
		self.result_cluster_s =kmeans.labels_
		np.save('result_cluster_s', self.result_cluster_s)

	def LS1_compressing(self):
		print("LS1_compressing")
		self.training_data_np = np.load('./training_data_np_save.npy')
		self.result_cluster_s = np.load('./result_cluster_s.npy')
		self.training_result_thrashhold = np.load('./training_result_thrashold.npy')
		self.tdnTrans = np.transpose(self.training_data_np)
		self.training_data_pca = []
		self.clusterLabel = []
		self.numberOfFisets = self.training_data_np.shape[1]
		self.training_data_with_cl = []
		pca = PCA(n_components=1)
		for i in range(0,14):
			result = np.where(self.result_cluster_s == i)
			if result[0].size == 0:
				continue
			#pca.fit(np.transpose(self.tdnTrans[result]))
			#r_pca = pca.transform(np.transpose(self.tdnTrans[result]))
			#print(pca.explained_variance_ratio_)
			#self.training_data_pca.append(r_pca)
			#cw = self.LS0_Cluster_Weighter(self.training_data_np.shape[1], result[0].size)
			for r in self.tdnTrans[result]:
				cl = self.LS1_CorrealtionWeighter(self.training_result_thrashhold, r)
			#self.training_data_with_cl.append({'cw':cw, 'data':self.training_data_pca})

		#np.array(self.training_data_pca).reshape(len(self.training_data_pca), len(self.training_data_pca[0]))
		print("LS1_compressing fin")

	def LS1_CorrealtionWeighter(self, trt, tdp):
		result  = stats.pearsonr(trt, tdp)
		return result
		print("LS1_CorrealtionWeighter")

	def LS0_Cluster_Weighter(self,totalFisets, subfisets):
		print("LS0_Cluster_Weighter")
		return subfisets/totalFisets

	def LS2_WeightingProductor(self):
		print("LS2_WeightingProductor")



sc = StockClassfication()
#sc.prepareData()
#sc.LLS0_VarianceMaximizer()
#sc.priceThrashhold()
sc.LS1_compressing()
