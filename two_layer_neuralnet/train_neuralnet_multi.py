# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import datetime
from two_layer_net import TwoLayerNet
from multi_layer_net import MultiLayerNet
from kdd_load_2classes import *
from dataset_class import *

# kdd_load_2classes.pyからTrain,Testデータを取得
kdd_label, kdd_data, kdd_type, kdd_label_test, kdd_data_test, kdd_type_test = kdd_load()

#dataset_class.pyを使用してデータを格納
#各クラスの一定割合をラベル付きデータにする
dataset = Dataset(train=(kdd_data, kdd_label),test=(kdd_data_test, kdd_label_test))
x_train, t_train = dataset.get_labeled_data()
x_test, t_test = dataset.get_test_data()

network = MultiLayerNet(input_size=122, hidden_size_01=50, hidden_size_02=50, hidden_size_03=50, output_size=2)

iters_num = 50000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch_1 = max(train_size / batch_size, 1)
iter_per_epoch = round(iter_per_epoch_1) #浮動小数点を実数に変更

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, test_acc_list, label='test_acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig("./Graph/{}.png".format(datetime.datetime.now()))
