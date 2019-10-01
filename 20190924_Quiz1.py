# 수정 필요 없음, 그러나 필요시 수정 가능

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
  

    
# 알고리즘이 정상적으로 작성되었다면 수정할 필요 없음
predict_centers_Lloyd, predict_labels_Lloyd = kmeans_Lloyd(X, 10)

from sklearn.cluster import KMeans
    
def kmeans_plus(X,n_cluster):
    
  kmeans = KMeans(n_cluster, init='k-means++').fit(X)
  labels = kmeans.labels_

  centers= kmeans.predict(X)
  
  
  return centers, labels

predict_centers_Plus, predict_labels_Plus = kmeans_plus(X, 10)

predict_centers_Lloyd.type()


### ??? 를 작성한다.

predict_centers_ =  predict_labels_Plus
fig, ax = plt.subplots(2, 5, figsize=(8, 3))

for axi, center in zip(ax.flat, predict_centers_):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


## ========================================================================
## kmeans 군집 결과는 실제 레이블에 매칭되지 않는다.
## 학습 된 각 클러스터 레이블을 실제 레이블과 일치시키는 과정이 필요하다.
## ========================================================================



labels = predict_labels_Lloyd

## 아래의 코드는 수정할 필요가 없다
from sklearn.metrics import accuracy_score
accuracy_score(Y,labels)

## ========================================================================
## kmeans 군집 결과는 실제 레이블에 매칭되지 않는다.
## 학습 된 각 클러스터 레이블을 실제 레이블과 일치시키는 과정이 필요하다.
## ========================================================================



labels = predict_labels_Plus

## 아래의 코드는 수정할 필요가 없다
from sklearn.metrics import accuracy_score
accuracy_score(Y,labels)


predict_labels_Lloydrk가 조금더 좋은 성능을 보였다.. 하지만 뭔가 잘못불러와서 측정한것 같다. 