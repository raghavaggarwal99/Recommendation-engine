import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("rag.csv")

features = ['Product','Sub-category','Category','Store Name']

for feature in features:
    df[feature] = df[feature].fillna('')

for feature in features:
    labelEncoder = LabelEncoder()
    labelEncoder.fit(df[feature])
    df[feature] = labelEncoder.transform(df[feature])

df= df.drop(['Store-Id','Date of Purchase', 'Qunatity','Phone','Bill No','Index'], axis=1)

# df.info()

X = np.array(df.drop(['Total Amount'], 1).astype(float))

Y = np.array(df['Total Amount'])

# print(Y)

kmeans = KMeans(n_clusters=4) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(int))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1
        # print(correct)

print(correct/len(X))


