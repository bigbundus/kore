import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('win_rates.p', 'rb') as f:
    win_rates = pickle.load(f)

wr_collect = []

for p1 in win_rates.keys():
    row = []
    for p2 in win_rates[p1].keys():
        print(win_rates[p1][p2])
        row.append(win_rates[p1][p2]/10.)
    wr_collect.append(row)

plt.figure()
plt.imshow(np.array(wr_collect))

plt.xticks([0,1,2,3,4], win_rates.keys())
plt.yticks([0,1,2,3,4], win_rates.keys())

plt.show()
