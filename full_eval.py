from kaggle_environments import make
import kaggle_environments
import pickle

def determine_winner(trace):
    last_frame = z[-1]
    players = last_frame[0]['observation']['players']
    p1_kore = players[0][0]
    p2_kore = players[1][0]
    print(p1_kore, p2_kore)

    # NOPE! need to catch ties correctly
    asdasdas (will break)
    if p1_kore > p2_kore:
        return 0
    else:
        return 1

#env = make("kore_fleets", debug=True)

agents = ["random", "miner", "do_nothing", "balanced", "attacker"]

win_counts = {}
runs_per_pairing = 10

for a1 in agents:
    win_counts[a1] = {}
    for b1 in agents:
        win_counts[a1][b1] = 0

        print(a1, " vs ", b1)

        for j in range(runs_per_pairing):
            env = make("kore_fleets", debug=True)
            z = env.run([a1, b1])

            p1_winner = 1 - determine_winner(z)

            win_counts[a1][b1] += p1_winner

print(win_counts)

with open('win_rates.p', 'wb') as f:
    pickle.dump(win_counts, f)
