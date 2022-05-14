from kaggle_environments import make
import kaggle_environments

env = make("kore_fleets", debug=True)

z = env.run(["random", "random"])

def determine_winner(trace):
    last_frame = z[-1]
    players = last_frame[0]['observation']['players']
    p1_kore = players[0][0]
    p2_kore = players[1][0]
    print(p1_kore, p2_kore)
    if p1_kore > p2_kore:
        return 0
    else:
        return 1

# can you win if you get eliminated but have a huge Kore stockpile?
print(determine_winner(z))
