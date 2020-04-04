import numpy as np
import pandas as pd
import collections
class KNNClassification(object):
    def __init__(self, k, p, weight='uniform'):
        self.k = k
        self.p = p
        self.weight = weight

    def fit(self, X, Y):
        '''
        :param x_train: matrix shape = (m, n)
        :param y_train: vector shape = (m, 1)
        :return: None
        '''
        self.X = X
        self.Y = Y
        self.m, self.n = X.shape

    def distance(self, z1, z2, p):
        return np.sum((z1-z2)**p)**(1/p)

    def k_nearest(self, z, k):
        Xd = {self.distance(self.X[i, :], z, self.p): self.Y[i] for i in range(self.m)}
        k_xd = sorted(Xd.keys(), reverse=True)[-k:]
        return {Xd[xdi]: xdi for xdi in k_xd}

    def most_frequent(self, knn):
        counter = 0
        num = knn[0]

        for i in set(knn):
            curr_frequency = knn.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i
        return num

    def predict(self, Z):
        result = []
        if self.weight is 'uniform':
            for z in Z:
                knn = self.k_nearest(z, self.k)
                output_values = list(knn.keys())
                pred = self.most_frequent(output_values)
                result.append(pred)

        elif self.weight is 'distance':
            result = None

        return result

def split_by_label(data_df, split_train):
    by_label = collections.defaultdict(list)

    for _, row in data_df.iterrows():
        by_label[row.Outcome].append(row.to_dict())
    final_list = []
    for _, item_list in sorted(by_label.items()):
        np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(split_train * n)
        for item in item_list[:n_train]:
            item['split'] = 'train'

        for item in item_list[n_train:]:
            item['split'] = 'test'
        final_list.extend(item_list)

    final_df = pd.DataFrame(final_list)
    return final_df

def accuracy_score(y_target, y_pred):
    return np.mean(y_target == y_pred)
file_data = 'data/diabetes.csv'

data_df = pd.read_csv(file_data)
data_df.dropna()
data = split_by_label(data_df, split_train=0.9)

X_train = data[data.split == 'train'].drop(['Outcome', 'split'], axis=1).to_numpy()
Y_train = data[data.split == 'train']['Outcome'].to_numpy()

X_test = data[data.split == 'test'].drop(['Outcome', 'split'], axis=1).to_numpy()
Y_test = data[data.split == 'test']['Outcome'].to_numpy()

knn = KNNClassification(k=10, p=2)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)

print("accuracy: ", accuracy_score(Y_test, prediction))