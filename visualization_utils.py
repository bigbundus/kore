import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import collections
import re

import kaggle_environments


def draw_board_from_obs(observation):
    plt.clf()
    plt.xlim((-0.5,20.5))
    plt.ylim((-0.5,20.5))

    # observation[0]["observation"]["kore"]
    kore_amounts = np.array(observation[0]["kore"])
    draw_kore_amounts(kore_amounts)

    color="blue"
    player_idx=0

    for shipyard_info in observation[0]["players"][player_idx][1].values():
        loc_idx, ships_count, existence = shipyard_info
        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)
        draw_shipyard(x-0.5,y-0.5, ships_count, existence, color)
        #excluded_xys.add((x,y))

    for fleet_info in observation[0]["players"][player_idx][2].values():
        loc_idx, kore_amount, ships_size, dir_idx, flight_plan = fleet_info
        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)
        draw_fleet(x-0.5,y-0.5,dir_idx,ships_size,kore_amount, color)
        draw_flight_plan(x-0.5,y-0.5,dir_idx,flight_plan,ships_size, color)
        #excluded_xys.add((x,y))

    color="red"
    player_idx=1

    for shipyard_info in observation[0]["players"][player_idx][1].values():
        loc_idx, ships_count, existence = shipyard_info
        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)
        draw_shipyard(x-0.5,y-0.5, ships_count, existence, color)
        #excluded_xys.add((x,y))

    for fleet_info in observation[0]["players"][player_idx][2].values():
        loc_idx, kore_amount, ships_size, dir_idx, flight_plan = fleet_info
        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)
        draw_fleet(x-0.5,y-0.5,dir_idx,ships_size,kore_amount, color)
        draw_flight_plan(x-0.5,y-0.5,dir_idx,flight_plan,ships_size, color)
        #excluded_xys.add((x,y))

    turn_num = observation[0]["step"]
    home_stored_kore = observation[0]["players"][0][0]
    away_stored_kore = observation[0]["players"][1][0]
    draw_statistics(turn_num, home_stored_kore, away_stored_kore)

def existence_to_production_capacity(existence):
    if existence >= 294: return 10
    if existence >= 212: return 9
    if existence >= 147: return 8
    if existence >= 97: return 7
    if existence >= 60: return 6
    if existence >= 34: return 5
    if existence >= 17: return 4
    if existence >= 7: return 3
    if existence >= 2: return 2
    return 1

def split_into_number_and_char(srr):
    # https://stackoverflow.com/q/430079/5894029
    arr = []
    for word in re.split('(\d+)', srr):
        try:
            num = int(word)
            arr.append(num)
        except ValueError:
            for c in word:
                arr.append(c)
    return arr

def draw_statistics(turn_num, home_stored_kore, away_stored_kore):
    plt.text(0-0.5, 21-0.5, f"Kore: {home_stored_kore:.0f}" , color="blue",
             horizontalalignment='left', verticalalignment='bottom')
    plt.text(21-0.5, 21-0.5, f"Kore: {away_stored_kore:.0f}" , color="red",
             horizontalalignment='right', verticalalignment='bottom')
    plt.text(21/2-0.5, 21-0.5, f"Turn: {turn_num:.0f}" , color="black",
             horizontalalignment='center', verticalalignment='bottom')

def draw_kore_amounts(kore_amounts, excluded_xys={}):
    for loc_idx,kore_amount in enumerate(kore_amounts):
        x,y = kaggle_environments.helpers.Point.from_index(loc_idx, 21)
        #print(x,y)
        color = "gainsboro"
        if kore_amount >= 20: color = "silver"
        if kore_amount >= 100: color = "gray"
        if kore_amount >= 500: color = "black"
        if (x,y) not in excluded_xys and kore_amount > 0:
            text = plt.text(x, y, int(kore_amount), color=color, fontsize=7,
                            horizontalalignment='center', verticalalignment='center')
            #text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='w', alpha=0.8)])

def draw_fleet(x,y,dir_idx,ships_size,kore_amount,color):
    mx,my = x+0.5, y+0.5

    icon_size = 0.4
    tip = (0, icon_size)
    left_wing = (icon_size/1.5, -icon_size)
    right_wing = (-icon_size/1.5, -icon_size)

    polygon = plt.Polygon([tip, left_wing, right_wing], color=color, alpha=0.3)
    transform = matplotlib.transforms.Affine2D().rotate_deg(270*dir_idx).translate(mx,my)
    polygon.set_transform(transform + plt.gca().transData)
    plt.gca().add_patch(
        polygon
    )

    text = plt.text(x+0.1, y+0.75, ships_size, color="purple",
                    horizontalalignment='left', verticalalignment='center')
    #text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])

    kore_amount = int(kore_amount)
    if kore_amount > 0:
        text = plt.text(x+0.1, y+0.25, kore_amount, color="grey",
                        horizontalalignment='left', verticalalignment='center')
        #text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='w', alpha=0.8)])

def extract_flight_plan(x,y,dir_idx,plan,endpoints=set()):
    dir_to_dxdy = [(0,1), (1,0), (0,-1), (-1,0)]  # NESW
    dcode_to_dxdy = {"N":(0,1), "E":(1,0), "S":(0,-1), "W":(-1,0)}
    dx,dy = dir_to_dxdy[dir_idx]

    plan = collections.deque(split_into_number_and_char(plan))

    cx,cy = x, y
    path = [(cx,cy)]
    construct = []
    first_move_complete = False

    while plan:
        if first_move_complete and (cx, cy) in endpoints:
            return path, construct, (cx, cy) == (x,y)
        first_move_complete = True
        word = plan.popleft()
        if type(word) == int:
            cx += dx
            cy += dy
            path.append((cx,cy))
            word -= 1
            if word > 0:
                plan.appendleft(word)
            continue
        if word == "C":
            construct.append((cx,cy))
            continue
        dx,dy = dcode_to_dxdy[word]
        cx += dx
        cy += dy
        path.append((cx,cy))

    is_cyclic = False
    for _ in range(30):
        if cx == x and cy == y:
            is_cyclic = True
        cx += dx
        cy += dy

    return path, construct, is_cyclic

def draw_flight_plan(x,y,dir_idx,plan,fleetsize,color):

    path, construct, is_cyclic = extract_flight_plan(x,y,dir_idx,plan)

    px = np.array([x for x,y in path]) + 0.5
    py = np.array([y for x,y in path]) + 0.5
    for ox in [-21,0,21]:
        for oy in [-21,0,21]:
            plt.plot(px+ox, py+oy, color=color, lw=np.log(fleetsize)**2/1.5, alpha=0.3, solid_capstyle='round')
    for x,y in construct:
        plt.scatter(x+0.5, y+0.5, s=100, marker="x", color=color)

def draw_shipyard(x,y,ships_size,existence,color):
    plt.text(x+0.5,y+0.5,"⊕", fontsize=23, color=color,
             horizontalalignment='center', verticalalignment='center', alpha=0.5)
    if ships_size > 0:
        text = plt.text(x+0.1, y+0.75, ships_size, color="purple",
                        horizontalalignment='left', verticalalignment='center')
        #text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
    text = plt.text(x+0.9, y+0.25, existence_to_production_capacity(existence), color="black",
                    horizontalalignment='right', verticalalignment='center')
    #text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='w', alpha=0.8)])
