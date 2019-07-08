import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("rag.csv")

features = ['Product','Sub-category','Category','Store Name']

def combine_features(row):
    return row['Product'] +" "+row['Sub-category']+" "+row["Category"]+" "+row["Store Name"]

# If null fill it with a blank
for feature in features:
    df[feature] = df[feature].fillna('')

df["Total Amount"]=df["Total Amount"].fillna('')

df["combined_features"] = df.apply(combine_features,axis=1)

#convert string to int
cv = CountVectorizer()

#includeing the total amount too
count_matrix = cv.fit_transform(df["combined_features"])

#cos all
cosine_sim = cosine_similarity(count_matrix)


def get_title_from_index(Index):
    return df[df.Index == Index]["Product"].values[0]

def get_index_from_title(product):
    return df[df.Product == product]["Index"].values[0]

food = "Citra Bliss"
index = get_index_from_title(food)
similar_products =  list(enumerate(cosine_sim[index]))
# print(similar_products)

sorted_order = sorted(similar_products,key=lambda x:x[1],reverse=True)[1:]

Dict={food: 1}

i=0
print("Most similar to the "+food+" are:\n")
for element in sorted_order:
    v=get_title_from_index(element[0])
    if v in Dict:
        # i=i+1
        continue
    else:
        print(v)
        Dict.update({v : 1})
        i=i+1
    if i>=5:
        break