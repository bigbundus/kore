from kaggle_environments.envs.kore_fleets.helpers import (
    Board,
    ShipyardAction
)

def simple_agent(obs, config):
    #model_observation = np.reshape(custom_env.build_observation(obs), [1, 21, 21, 4])
    #action_space = model.predict(model_observation)


    # Iterate over shipyards
    board = Board(obs, config)
    #custom_env.board = board
    if board.current_player.shipyards:
        board.current_player.shipyards[0].next_action = ShipyardAction.spawn_ships(1)#custom_env.match_action(action_space)

    return board.current_player.next_actions
