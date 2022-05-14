import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    ShipyardAction
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, linear_input_size)
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)


# observation to model_input
def observation_to_model_input(obs, config):
    kore_layer = [x/100. for x in obs['kore']]
    kore_layer = np.array(kore_layer).reshape(21,21)

    board = Board(obs, config)
    fleets = [fleet for _, fleet in board.fleets.items()]
    shipyards = [shipyard for _, shipyard in board.shipyards.items()]
    fleet_layer = np.zeros((21,21))
    shipyard_layer = np.zeros((21,21))

    for fleet in fleets:
        # - Get the position of the fleet
        x, y = fleet.position.x, fleet.position.y
        # - Get the size of the fleet
        size = fleet.ship_count
        # - Check if the fleet is an enemy fleet
        if fleet.player != board.current_player:
            player = -1
        else:
            player = 1
        # - Set the fleet layer to the size of the fleet
        fleet_layer[x, y] = player * size/20.

    # - Iterate over shipyards, getting its position and size
    for shipyard in shipyards:
        #print(shipyard.position)
        # - Get the position of the shipyard
        x, y = shipyard.position.x, shipyard.position.y
        # - Get the size of the shipyard
        #size = shipyard.ship_count
        # - Check if the shipyard is an enemy shipyard
        if shipyard.player != board.current_player:
            player = -1
        else:
            player = 1
        # - Set the fleet layer to the size of the shipyard
        shipyard_layer[x, y] = player

    model_input = np.stack([kore_layer, fleet_layer, shipyard_layer], -1)
    return model_input

# model output to actions
def model_output_to_actions(model_output, shipyard):
    '''
    also NOTE - have to do this for each shipyard seperately...

    model output should just be a single integer
    0: do nothing
    1: spawn max ships

    2: launch all ships north
    3: launch all ships east
    4: launch all ships south
    5: launch all ships west

    6: launch HALF ships north
    7: launch HALF ships east
    8: launch HALF ships south
    9: launch HALF ships west

    10: launch E-S attack (S9E)
    11: launch S-E attack (E9S)
    '''

    if model_output == 0:
        next_action = None

    max_spawnable = shipyard.max_spawn

    if model_output == 1:
        next_action = ShipyardAction.spawn_ships(max_spawnable) # should actually be max...

    nb_of_ships = max(1, shipyard.ship_count)

    if model_output == 2:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, 'N')
    if model_output == 3:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, 'E')
    if model_output == 4:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, 'S')
    if model_output == 5:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, 'W')



    half_ships = max(1, int(nb_of_ships/2))

    if model_output == 6:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(half_ships, 'N')
    if model_output == 7:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(half_ships, 'E')
    if model_output == 8:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(half_ships, 'S')
    if model_output == 9:
        next_action = ShipyardAction.launch_fleet_with_flight_plan(half_ships, 'W')

    if model_output == 10:
        if nb_of_ships > 2:
            next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, 'S9E')
        else:
            next_action = None
    if model_output == 11:
        if nb_of_ships > 2:
            next_action = ShipyardAction.launch_fleet_with_flight_plan(nb_of_ships, 'E9S')
        else:
            next_action = None


    return next_action


# define the model

# optimize model

# training loop
