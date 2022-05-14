from kaggle_environments import make
import random
from visualization_utils import *
from torch_agent import *

from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    ShipyardAction
)

import pickle

kor_env = make("kore_fleets", debug=True)


env_config = kor_env.configuration






BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:

        with torch.no_grad():
            #print("using policy net!")
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

screen_height, screen_width = 21, 21

# Get number of actions from gym action space
n_actions = 12

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(20000)

steps_done = 0

def optimize_model():
    #print("OPTIMIZING NETWORK!")
    if len(memory) < BATCH_SIZE:
        #print("too early, memory too small")
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def tensorize_state(model_input):
    state = np.moveaxis(model_input,-1,0).astype(np.float32)
    state = np.expand_dims(state,0)
    tensor_state = torch.from_numpy(state)
    return tensor_state

collect_episodes = []
num_episodes = 100
for i_episode in range(num_episodes):
    episode_record = []

    enemy_agent = random.choice(["random", "do_nothing", "do_nothing"]) #"miner",
    print("Episode ", i_episode)
    print("Enemy Agent:", enemy_agent)
    # Initialize the environment and state
    env = kor_env.train([None, enemy_agent])
    observation = env.reset()
    last_state = tensorize_state(observation_to_model_input(observation, env_config))
    current_state = tensorize_state(observation_to_model_input(observation, env_config))

    for t in count():

        #print(current_state.shape)
        action_int = select_action(current_state)

        board = Board(observation, env_config)

        #actions = []

        for shipyard in board.current_player.shipyards:
            next_action = model_output_to_actions(action_int.item(), shipyard)
            shipyard.next_action = next_action
            #actions.append(next_action)

        observation, old_reward, done, info = env.step(board.current_player.next_actions)

        last_state = current_state
        current_state = tensorize_state(observation_to_model_input(observation, env_config))

        memory.push(last_state,
            action_int,
            current_state,
            torch.from_numpy(np.array([old_reward]).astype(np.float32)).cuda())

        optimize_model()
        if done:
            #episode_durations.append(t + 1)

            episode_record.append(t)
            episode_record.append(observation['players'][0][0])
            episode_record.append(observation['players'][1][0])
            #plot_durations()
            print("Episode done! Steps: ", t+1)
            print("Scores: training agent:", observation['players'][0][0], "Enemy agent:", observation['players'][1][0])
            print("----------")

            collect_episodes.append(episode_record)
            break

        #draw_board_from_obs(observation)
        #print(t, old_reward)
        #plt.pause(0.1)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

for e in collect_episodes:
    print(e)

with open('episode_scores_1k.p', 'wb') as f:
    pickle.dump(collect_episodes, f)

"""
plt.figure(figsize=(9,9))

for model_output in [1,1,1,1,1,1,1,4]:
    board = Board(observation, env_config)

    for shipyard in board.current_player.shipyards:
        next_action = model_output_to_actions(model_output, shipyard)
        shipyard.next_action = next_action

    observation, old_reward, done, info = env.step(board.current_player.next_actions)
    draw_board_from_obs(observation)

    plt.pause(2)

plt.show()
"""
