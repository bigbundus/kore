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

agents = {"random": random_agent,
            "miner": miner_agent,
            "do_nothing": do_nothing_agent,
            "balanced": balanced_agent,
            "attacker": attacker_agent}
            
env = kor_env.train([None, "random"])

env_configuration= kor_env.configuration


plt.figure(figsize=(10,10))


#val = input("enter something!")
#print(val)


def get_player_action(board):
    # actions
    # do nothing, spawn ships, launch fleet with flightplan
    player_input = -1

    while player_input not in [0,1,2]:
        print("-- Actions --")
        print("0: Do nothing")
        print("1: Spawn ships")
        print("2: Launch fleet with flight plan")
        player_input = int(input("Enter Action Type: "))

    if player_input == 0:
        next_action = None

    if player_input == 1:
        nb_of_ships = int(input("Enter number of ships to spawn: "))
        next_action = ShipyardAction.spawn_ships(nb_of_ships)

    if player_input == 2:
        nb_of_ships = int(input("Enter number of ships to launch: "))
        flight_plan = input("Enter flight plan: ")
        next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, flight_plan)


    board.current_player.shipyards[0].next_action = next_action
    print("--------------")
    #return 0



observation = env.reset()


for i in range(8):

    board = Board(observation, env_configuration)
    get_player_action(board)



    # need to assign actions to shipyards!
    #print(action)
    #from step we get: [raw_observation, old_reward, done, info]
    observation, old_reward, done, info = env.step(board.current_player.next_actions)
    #print(observation[0])

    # need to check if game over!!!
    # and end loop if so

    draw_board_from_obs(observation)

    plt.pause(0.1)


    p1_nb_shipyards = len(observation['players'][0][1].keys())
    p2_nb_shipyards = len(observation['players'][1][1].keys())
    #this isnt correct! could have no shipyard but still have enough kore to make a new one with a ship
    # actually need 50 ships,not kore to convert to new shipyard
    # need to check kore amount
    # actually, just check the done return from step!

    #if p1_nb_shipyards == 0 or p2_nb_shipyards == 0
    if done:
        print("Game over!")
        break
    #else:
        #print("player 1 shipyards:", p1_nb_shipyards)
        #print("player 2 shipyards:", p2_nb_shipyards)


plt.show()
