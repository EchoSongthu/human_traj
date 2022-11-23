import pickle
import pdb
import json
from config import *
import numpy as np
max_grid = args.max_grid

path = '/data2/songyiwen/human_traj_/dataset/split/train.pkl'
split_data = pickle.load(open(path, 'rb'))
test_data = []

for id_,i in enumerate(split_data):
    test_data.append([])
    for j in i:
        for p in j:
            grid_id = p[0] * max_grid + p[1] +1
            test_data[id_].append(grid_id)
test_data = np.array(test_data)
pdb.set_trace()

path = '/data2/songyiwen/human_traj_/dataset/split/test_data.pkl'
pickle.dump(test_data, open(path,'wb'))
print("done")