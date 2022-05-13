from kaggle_environments import make
import kaggle_environments

import numpy as np
import matplotlib.pyplot as plt

from visualization_utils import *

from kaggle_environments.envs.kore_fleets.helpers import (
    ShipyardAction
)

kor_env = make("kore_fleets", debug=True)
#print(env.name, env.version)

env = kor_env.train([None, "random"])
observation = env.reset()
observation = env.step([0])
# from step we get: [raw_observation, old_reward, done, info]
# how is reward calculated? is it kore delivered?



print(env)
#print(ShipyardAction.launch_fleet_with_flight_plan(1,'N'))
#print(type(ShipyardAction.launch_fleet_with_flight_plan(1,'N')))
#launch_fleet_in_direction
#launch_fleet_with_flight_plan
#spawn_ships
#do nothing

'''

plt.figure(figsize=(10,10))



for i in range(50):
    observation = env.step([0])
    print(observation[0])

    # need to check if game over!!!
    # and end loop if so

    draw_board_from_obs(observation)

    plt.pause(0.1)
    val = input("enter something!")

    p1_nb_shipyards = len(observation[0]['players'][0][1].keys())
    p2_nb_shipyards = len(observation[0]['players'][1][1].keys())

    print(val)
    if p1_nb_shipyards == 0 or p2_nb_shipyards == 0:
        print("Game over!")
        break
    else:
        print("player 1 shipyards:", p1_nb_shipyards)
        print("player 2 shipyards:", p2_nb_shipyards)


plt.show()
'''
