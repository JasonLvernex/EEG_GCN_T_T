import pandas
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# 读取测试集和测试集
train_data = pandas.read_csv("./data/biotacsp/raw/train_dataset.csv")#,header=None
test_data = pandas.read_csv("./data/biotacsp/raw/val_dataset.csv")

# clf = svm.SVC(kernel='linear')  # 参数 kernel 为线性核函数
clf = svm.LinearSVC(C=50,max_iter=10000)#C越大，惩罚越大，越不容易出现分类错误的情况 ##数据量较大时，使用linearSVC速度更快
# clf = svm.SVC(C=10,kernel='rbf')
predictors_o = ['FMD','FR','SSC','SSI','HjorthActivity',
                    'HjorthComplexity','HjorthMobility','RenyiEntropy','TsallisEntropy','BandPowerAlpha','BandPowerBeta',
                    'BandPowerDelta','BandPowerGamma','BandPowerTheta','RatioBandPowerAlphaBeta','Kurtosis','Skewness',
                    'Maxpeak','RMS','MV',' WAMP','ZC','VAR']
# predictors_o = ['FMD','MV','HjorthComplexity']
predictors=[]
for i in predictors_o:
    for j in range(1,64):
        predictors.append(i+str(j))
print(f'sizeof_pre:{len(predictors)}')
trian_predictors = (train_data[predictors])

train_target = train_data['label']
# 训练分类器
clf.fit(trian_predictors, train_target)
# b = clf.intercept_
# w = clf.coef_
# print('各类别各有多少个支持向量', clf.n_support_)
# print('各类型的支持向量在训练样本中的索引', clf.support_)
# print('各类所有支持向量', clf.support_vectors_)
# print('支持向量的alpha值', clf.dual_coef_)

predictions = []
test_predictions = clf.predict(test_data[predictors])
score_test = clf.score(test_data[predictors], test_data['label'])
print('SVM Score: {}'.format(score_test))

# predictions.append(test_predictions)
# predictions = np.concatenate(predictions, axis=0)
# predictions[predictions > 0.5] = 1
# predictions[predictions <= 0.5] = 0
# accuracy = sum(predictions[predictions == test_data['label']]) / len(predictions)
# print('accuracy', end=' = ')
# print(accuracy * 100, end='')
# print('%')

# fig = plt.figure()
# ax=fig.add_subplot(111)
# cm_dark = mpl.colors.ListedColormap(['g', 'r'])
# ax.scatter(np.array(trian_predictors)[:, 0], np.array(trian_predictors)[:, 1], c=np.array(train_target).squeeze(), cmap=cm_dark, s=30)
# #决策平面
# x0 = np.arange(-2.0, 12.0, 0.1)
# x1 = (-w[0][0]*x0-b)/w[0][1]
# ax.plot(x0, x1.reshape(-1, 1))
#
# #w0*x0+w1*x1+b=y
# #画间隔平面
# pos0 = np.arange(-2.0, 12.0, 0.1)
# pos1 = (1-w[0][0]*pos0 - b)/w[0][1]
# ax.plot(pos0, pos1.reshape(-1, 1), color='green')
#
# neg0 = np.arange(-2.0, 12.0, 0.1)
# neg1 = (-1-w[0][0]*neg0-b)/w[0][1]
# ax.plot(neg0, neg1.reshape(-1, 1), color='green')
#
# ax.axis([-100,100,-100,100])#([-2, 12, -8,6])
# plt.show()