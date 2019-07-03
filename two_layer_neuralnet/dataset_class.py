import numpy as np
import math
class Dataset():
    def __init__(self, train, test, num_classes=2, percent=1.0):
        self.data_train, self.labels_train = train
        self.data_test, self.labels_test = test
        self.num_classes = num_classes
        self.percent = percent
        indices = np.arange(0, len(self.data_train))
        np.random.shuffle(indices)
        indices_u = []
        indices_l = []
        indices_test= []
        counts = [0] * num_classes
        num_per_class =[0] * num_classes
        totalnum_per_class = [0] * num_classes

        #testに各クラスごとの枚数を格納
        for i in range(num_classes) :
            totalnum_per_class[i] = np.sum(self.labels_train == i)

        #testに入っている各クラスごとの総数から何枚にラベルがつくのかを算出
        for i in range(num_classes) :
            num_per_class[i] = math.floor( percent *  totalnum_per_class[i] )

        #各ラベルに対して，上限枚数に達するまでデータを追加していく，残りはラベルなしデータにする
        for index in indices:
            label = self.labels_train[index]
            if counts[label] < num_per_class[label]:
                counts[label] += 1
                indices_l.append(index)
                continue
            indices_u.append(index)
        
        #テストはすべて使用
        indices_test = np.arange(0, len(self.data_test))

        #変換とシャッフル
        self.indices_l = np.asarray(indices_l)
        self.indices_u = np.asarray(indices_u)
        self.indices_test = np.asarray(indices_test)
        self.shuffle()

    def get_labeled_data(self):
        return self.data_train[self.indices_l], self.labels_train[self.indices_l]

    def get_unlabeled_data(self):
        return self.data_train[self.indices_u], self.labels_train[self.indices_u]

    def get_test_data(self):
        return self.data_test, self.labels_test

    def get_num_labeled_data(self):
        return len(self.indices_l)

    def get_num_unlabeled_data(self):
        return len(self.indices_u)

    def shuffle(self):
        np.random.shuffle(self.indices_l)
        np.random.shuffle(self.indices_u)
        np.random.shuffle(self.indices_test)