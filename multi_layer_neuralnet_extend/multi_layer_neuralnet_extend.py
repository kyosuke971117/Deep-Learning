# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import time
#from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from kdd_load_2classes import *
from dataset_class import *
from common.optimizer import *

dt = datetime.datetime.now()
second = dt.second
minute = dt.minute
hour = dt.hour
day = dt.day
month = dt.month
year = dt.year

#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

print("Loading data…")
# kdd_load_2classes.pyからTrain,Testデータを取得
kdd_label, kdd_data, kdd_type, kdd_label_test, kdd_data_test, kdd_type_test = kdd_load()
print("Finished Loading data…")

# 学習データを削減
#x_train = x_train[:1000]
#t_train = t_train[:1000]

print("Labeling data…")
#dataset_class.pyを使用してデータを格納
#各クラスの一定割合をラベル付きデータにする
dataset = Dataset(train=(kdd_data, kdd_label),test=(kdd_data_test, kdd_label_test),percent=1.0)
x_train, t_train = dataset.get_labeled_data() 
x_test, t_test = dataset.get_test_data() 
print("Finished Labeling data…")

#念のため、データの次元数の確認
print("x_train,t_train dimension_number")
print(x_train.shape,t_train.shape)
print("x_test,t_test dimension_number")
print(x_test.shape,t_test.shape)

# 高速化のため訓練データの削減
#x_train = x_train[:56678]
#t_train = t_train[:56678]

#optimizerを選択
optimizers = Adam()

network = MultiLayerNetExtend(input_size=122, hidden_size_list=[1000,1000], output_size=2,
                            activation='relu', weight_init_std=0.01,
                            weight_decay_lambda=0, use_dropout=True, dropout_ration=0.5,
                            use_batchnorm=False)

#ハイパーパラメータ
iters_num = 73554 #イテレーションの回数
batch_size = 512
learning_rate = 0.00001 #学習率

train_size = x_train.shape[0] #x_trainの行数を当てはめる

#格納するリストの作成
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch_float = max(train_size / batch_size, 1) #1エポック回すのに何イテレーションかかるか
iter_per_epoch = round(iter_per_epoch_float) #浮動小数点を実数に変更

#平均計算をするために使用されるfloat型
count = 0.0
train_sum = 0.0
test_sum = 0.0

#処理前の時刻
t1 = time.time()

for i in range(iters_num): #イテレーションの回数分for文を回している
    batch_mask = np.random.choice(train_size, batch_size) #batch_size分のbatch_size行列の乱数を作成
    x_batch = x_train[batch_mask] #x_subsetの作成
    t_batch = t_train[batch_mask] #t_subsetの作成
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) 

    optimizers.update(network.params, grad)

    # 更新
    #for key in ('W1', 'b1', 'W2', 'b2'):
        #network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch) #損失を計算
    train_loss_list.append(loss) #損失リストに格納する

    if i % iter_per_epoch == 0: #イテレーションの回数分回している中で、iがiter_per_epochの倍数の時に以下を実行
        #それぞれの認識精度を計算
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        #認識精度のリストに格納する
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        #すべてのtrain,testに対する認識精度を_sumに与える
        train_sum += train_acc
        test_sum += test_acc
        #実行した回数をカウントする
        count += 1.0
        print("count:",count,": train_acc:",train_acc," test_acc:",test_acc)
        
#処理後の時刻
t2 = time.time()
#経過時間
elapsed_time = t2-t1

#グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, test_acc_list, label='test_acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.4, 1.0)
plt.legend(loc='best')
plt.savefig("./Graph_multi_layer_neuralnet/{}-{}-{}-{}:{}:{}.png".format(year,month,day,hour,minute,second))

#出力結果
print("\n#######result#######")
print("\naverage_train_acc")
print(train_sum/count) #trainデータの認識精度の平均を出力
print("average_test_acc")
print(test_sum/count) #testデータの認識精度の平均を出力
print("\nmax_train_acc")
print(max(train_acc_list)) #trainデータの認識精度の最高値を出力
print("max_test_acc")
print(max(test_acc_list)) #testデータの認識精度の最高値を出力
print("\nmin_train_acc")
print(min(train_acc_list)) #trainデータの認識精度の最小値を出力
print("min_test_acc")
print(min(test_acc_list)) #testデータの認識精度の最小値を出力
print("\nelapsed_time")
print(elapsed_time) #経過時間を出力

#出力結果をcsvに書き込む
with open('csv/writer_acc.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow([train_sum/count,test_sum/count,max(train_acc_list),max(test_acc_list),min(train_acc_list),min(test_acc_list)])


'''
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
'''


'''def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list
    '''


'''3.グラフの描画==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()
'''