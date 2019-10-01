# ���� �ʿ� ����, �׷��� �ʿ�� ���� ����

from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
Y = digits.target


digits.data.shape

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import pairwise_distances_argmin

def kmeans_Lloyd(X, n_clusters, rseed=10):
  
  rng = np.random.RandomState(rseed)
  i = rng.permutation(X.shape[0])[:n_clusters]
  centers = X[i]
  
  while True:
    labels = pairwise_distances_argmin(X,centers)
    new_centers = np.array([X[labels ==i].mean(0)
                           for i in range(n_clusters)])
    if np.all(centers == new_centers):
      break
    centers = new_centers
  
  return centers, labels
  

    
# �˰����� ���������� �ۼ��Ǿ��ٸ� ������ �ʿ� ����
predict_centers_Lloyd, predict_labels_Lloyd = kmeans_Lloyd(X, 10)

from sklearn.cluster import KMeans
    
def kmeans_plus(X,n_cluster):
    
  kmeans = KMeans(n_cluster, init='k-means++').fit(X)
  labels = kmeans.labels_

  centers= kmeans.predict(X)
  
  
  return centers, labels

predict_centers_Plus, predict_labels_Plus = kmeans_plus(X, 10)

predict_centers_Lloyd.type()


### ??? �� �ۼ��Ѵ�.

predict_centers_ =  predict_labels_Plus
fig, ax = plt.subplots(2, 5, figsize=(8, 3))

for axi, center in zip(ax.flat, predict_centers_):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


## ========================================================================
## kmeans ���� ����� ���� ���̺� ��Ī���� �ʴ´�.
## �н� �� �� Ŭ������ ���̺��� ���� ���̺�� ��ġ��Ű�� ������ �ʿ��ϴ�.
## ========================================================================



labels = predict_labels_Lloyd

## �Ʒ��� �ڵ�� ������ �ʿ䰡 ����
from sklearn.metrics import accuracy_score
accuracy_score(Y,labels)

## ========================================================================
## kmeans ���� ����� ���� ���̺� ��Ī���� �ʴ´�.
## �н� �� �� Ŭ������ ���̺��� ���� ���̺�� ��ġ��Ű�� ������ �ʿ��ϴ�.
## ========================================================================



labels = predict_labels_Plus

## �Ʒ��� �ڵ�� ������ �ʿ䰡 ����
from sklearn.metrics import accuracy_score
accuracy_score(Y,labels)


predict_labels_Lloydrk�� ���ݴ� ���� ������ ������.. ������ ���� �߸��ҷ��ͼ� �����Ѱ� ����. 