# if __name__ == '__main__':
import pandas as pd
import csv
import numpy as np
import scipy
from scipy.sparse.linalg import svds, eigs
data = []

with open("ml-100k/u.data") as csvfile:
    spamreader = csv.reader(csvfile, delimiter="\t")
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)
num_users = max(data[:,0])+1
num_items = max(data[:,1])+1

# num_observations = 100,000
# num_users = 943, indexed 0,...,942
# num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]
movie_list = {}
for i in train:
    m_id = i[1]
    if m_id in movie_list.keys():
        value = movie_list.get(m_id)
        movie_list.update({m_id: value + 1})
    else:
        movie_list.update({m_id: 1})

movie_avg = np.zeros((len(movie_list.keys()), 2))
movie_ids = list(movie_list.keys())
for i in range(len(movie_ids)):
    m = movie_ids[i]
    movie_avg[i, 0] = m
    movie_avg[i, 1] = train[train[:, 1] == m][:, 2].mean()



tilta_R = np.zeros((num_items, num_users))

for i in range(num_items):
    for j in range(num_users):

        user = train[train[:, 0] == j]
        rat = user[user[:, 1] == i]
        if len(rat) != 0:
            rat = rat[0][2]
            tilta_R[i, j] = rat
        else:
            tilta_R[i, j] = 0

    print(i, j, tilta_R[i, j])

# pd.DataFrame(tilta_R).to_csv("tilta_R.csv")


print("Success")

tilta_R2 = pd.read_csv("tilta_R.csv").to_numpy()[:, 1:]

k = 100
u, s, vt = svds(tilta_R2, k=k)
S = np.zeros((k, k))
np.fill_diagonal(S, s)

u.shape
S.shape

vt.shape

A = u @ S @ vt
