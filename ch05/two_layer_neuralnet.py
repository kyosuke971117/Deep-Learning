# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
from two_layer_net import TwoLayerNet
from multi_layer_net import MultiLayerNet
from kdd_load_2classes_ch import *
from dataset_class_ch import *
from common.optimizer import *

dt = datetime.datetime.now()
second = dt.second
minute = dt.minute
hour = dt.hour
day = dt.day
month = dt.month
year = dt.year

# kdd_load_2classes.pyからTrain,Testデータを取得
kdd_label, kdd_data, kdd_type, kdd_label_test, kdd_data_test, kdd_type_test = kdd_load()

#dataset_class.pyを使用してデータを格納
#各クラスの一定割合をラベル付きデータにする
dataset = Dataset(train=(kdd_data, kdd_label),test=(kdd_data_test, kdd_label_test))
x_train, t_train = dataset.get_labeled_data() #113375行122列
x_test, t_test = dataset.get_test_data() #22544行122列

#optimizerを選択
optimizers = Adam()

network = TwoLayerNet(input_size=122, hidden_size=50, output_size=2)

#ハイパーパラメータ
iters_num = 50000 #イテレーションの回数
batch_size = 1024
learning_rate = 0.1 #学習率

train_size = x_train.shape[0] #x_trainの行数を当てはめる#113375

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch_1 = max(train_size / batch_size, 1) #1エポック回すのに何イテレーションかかるか#1133.75
iter_per_epoch = round(iter_per_epoch_1) #浮動小数点を実数に変更#1134

count = 0.0
train_sum = 0.0
test_sum = 0.0

for i in range(iters_num): #イテレーションの回数分for文を回している
    batch_mask = np.random.choice(train_size, batch_size) #batch_size分のbatch_size行1列の1〜113375の乱数を作成
    x_batch = x_train[batch_mask] #x_trainに対応
    t_batch = t_train[batch_mask] #t_trainに対応
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) 

    optimizers.update(network.params, grad)

    # 更新
    #for key in ('W1', 'b1', 'W2', 'b2'):
        #network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0: 
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_sum += train_acc
        test_sum += test_acc
        count += 1.0
        print("count:",count,": train_acc:",train_acc," test_acc:",test_acc)

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, test_acc_list, label='test_acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.4, 1.0)
plt.legend(loc='best')
plt.savefig("./Graph_two_layer_neuralnet/{}-{}-{}-{}:{}:{}.png".format(year,month,day,hour,minute,second))

#出力結果
print("\n###result###\n\naverage_train_acc")
print(train_sum/count) #trainデータの認識精度の平均を出力
print("\naverage_test_acc")
print(test_sum/count) #testデータの認識精度の平均を出力
print("\nmax_train_acc")
print(max(train_acc_list)) #trainデータの認識精度の最高値を出力
print("\nmax_test_acc")
print(max(test_acc_list)) #testデータの認識精度の最高値を出力

#出力結果をcsvに書き込む
with open('csv/writer_acc.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow([train_sum/count,test_sum/count,max(train_acc_list),max(test_acc_list)])
