import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('episode_scores_5k.p', 'rb') as f:
    episodes = pickle.load(f)

print(len(episodes))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

steps = [x[0] for x in episodes]
p1_scores = moving_average([x[1] for x in episodes], 20)
p2_scores = moving_average([x[2] for x in episodes], 20)

plt.figure()
plt.plot(p1_scores)
plt.plot(p2_scores)
plt.show()
