# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from two_layer_net import TwoLayerNet
from kdd_load_2classes_ch import *
from dataset_class_ch import *

# kdd_load_2classes.pyからTrain,Testデータを取得
kdd_label, kdd_data, kdd_type, kdd_label_test, kdd_data_test, kdd_type_test = kdd_load()

#dataset_class.pyを使用してデータを格納
#各クラスの一定割合をラベル付きデータにする
dataset = Dataset(train=(kdd_data, kdd_label),test=(kdd_data_test, kdd_label_test))
x_train, t_train = dataset.get_labeled_data()
x_test, t_test = dataset.get_test_data()

network = TwoLayerNet(input_size=122, hidden_size=50, output_size=2)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))