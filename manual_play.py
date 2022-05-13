from kaggle_environments import make
import kaggle_environments
from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    ShipyardAction
)

import numpy as np
import matplotlib.pyplot as plt

from visualization_utils import *



kor_env = make("kore_fleets", debug=True)
#print(env.name, env.version)

env = kor_env.train([None, "random"])

env_configuration= kor_env.configuration


plt.figure(figsize=(10,10))


#val = input("enter something!")
#print(val)


def get_player_action(board):
    #print()
    player_input = input("Enter Action Type: ")
    board.current_player.shipyards[0].next_action = ShipyardAction.spawn_ships(1)
    #return 0



observation = [env.reset()]


for i in range(8):

    board = Board(observation[0], env_configuration)
    get_player_action(board)



    # need to assign actions to shipyards!
    #print(action)

    observation = env.step(board.current_player.next_actions)
    #print(observation[0])

    # need to check if game over!!!
    # and end loop if so

    draw_board_from_obs(observation)

    plt.pause(0.1)


    p1_nb_shipyards = len(observation[0]['players'][0][1].keys())
    p2_nb_shipyards = len(observation[0]['players'][1][1].keys())


    if p1_nb_shipyards == 0 or p2_nb_shipyards == 0:
        print("Game over!")
        break
    else:
        print("player 1 shipyards:", p1_nb_shipyards)
        print("player 2 shipyards:", p2_nb_shipyards)


plt.show()
