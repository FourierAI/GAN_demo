
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
import pandas as pd
import matplotlib

import seaborn as sns


# 设置绘图大小
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc("font", family='Songti SC')
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 300)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
# 文件名
BCG_path = "/Users/gyw/Desktop/基于睡眠检测的闭环自适应分类/BCG-ECG/GAN-input/BCG_h(t)_self(前30000 测试用）/"
ECG_path = "/Users/gyw/Desktop/基于睡眠检测的闭环自适应分类/BCG-ECG/GAN-input/ECG_h(t)/"

# read csv
def read_csv(x):
	N=[]
	M = pd.read_csv(x, encoding='utf-8', header=None,index_col=None)
	M=M.values
	return M


def pearsonrSim(x, y):
	'''
	皮尔森相似度
	'''
	return pearsonr(x, y)[0]


def spearmanrSim(x, y):
	'''
	斯皮尔曼相似度
	'''
	return spearmanr(x, y)[0]


def kendalltauSim(x, y):
	'''
	肯德尔相似度
	'''
	return kendalltau(x, y)[0]


def cosSim(x, y):
	'''
	余弦相似度计算方法
	'''
	tmp = sum(a * b for a, b in zip(x, y))
	non = np.linalg.norm(x) * np.linalg.norm(y)
	return round(tmp / float(non), 3)


def eculidDisSim(x, y):
	'''
	欧几里得相似度计算方法
	'''
	return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
	'''
	曼哈顿距离计算方法
	'''
	return sum(abs(a - b) for a, b in zip(x, y))


def minkowskiDisSim(x, y, p):
	'''
	明可夫斯基距离计算方法
	'''
	sumvalue = sum(pow(abs(a - b), p) for a, b in zip(x, y))
	tmp = 1 / float(p)
	return round(sumvalue ** tmp, 3)


def MahalanobisDisSim(x, y):
	'''
	马氏距离计算方法
	'''
	npvec1, npvec2 = np.array(x), np.array(y)
	npvec = np.array([npvec1, npvec2])
	sub = npvec.T[0] - npvec.T[1]
	inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
	return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))

def main():
	beat_sum = 0

	# datasets = { '01b', '02a', '02b', '03', '04', '14', "16", "32", "37", "41", "45", "48", "59", "60", "61",
	# 			"66", "67x"}
	datasets = {"01b"}
	for name in datasets:
		# 文件名
		BCG_file = BCG_path + "slp" + str(name) + ".csv"
		ECG_file = ECG_path + "slp" + str(name) + ".csv"

		BCG=read_csv(BCG_file)[0:114]
		ECG=read_csv(ECG_file)[0:114]
		#abs
		for i in np.arange(0,114,5):
			BCG_chunk=BCG[i:i+5].flatten()
			ECG_chunk=ECG[i:i+5].flatten()
			plt.subplot(211)
			plt.plot(BCG_chunk)
			plt.subplot(212)
			plt.plot(ECG_chunk)
			plt.show()
			Beat_Comparision = np.array([BCG_chunk, ECG_chunk])

			# 计算协方差矩阵

			Single_cov=np.cov(Beat_Comparision)
			Single_corrcoef=np.corrcoef(Beat_Comparision)
			print(Single_cov)
			print(Single_corrcoef)
			figure, ax = plt.subplots(figsize=(5,5))
			plt.subplots(figsize=(5,5))  # 设置画面大小
			sns.heatmap(Single_corrcoef,annot=True, vmax=1, square=True, cmap="Blues",ax=ax)
			figure.show()

			sns.heatmap(Single_cov , square=True, annot=True, )
			figure.show()



if __name__ == '__main__':
	main()
