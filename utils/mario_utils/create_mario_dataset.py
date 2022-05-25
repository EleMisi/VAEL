import json
import math
import os
from itertools import product

import numpy as np
from PIL import Image
from matplotlib import use
from numpy import random
# d = Manhattan distance
from tqdm import tqdm

import const_define as cd

# Disable canvas visualization
use('Agg')

ICONS = {
    'glass': f'{cd.PROJECT_DIR}/data/mario_icons/glass.png',
    'flowers1': f'{cd.PROJECT_DIR}/data/mario_icons/flowers1.png',
    'flowers2': f'{cd.PROJECT_DIR}/data/mario_icons/flowers2.png',
    'brick': f'{cd.PROJECT_DIR}/data/mario_icons/brick.png',
    'brick2': f'{cd.PROJECT_DIR}/data/mario_icons/brick2.png',
    'brick3': f'{cd.PROJECT_DIR}/data/mario_icons/brick3.png',
    'concrete': f'{cd.PROJECT_DIR}/data/mario_icons/concrete.png',
    'wood': f'{cd.PROJECT_DIR}/data/mario_icons/wood.png',
    'white_panel': f'{cd.PROJECT_DIR}/data/mario_icons/white_panel.png',
    'green_panel': f'{cd.PROJECT_DIR}/data/mario_icons/green_panel.png',

    'lava': f'{cd.PROJECT_DIR}/data/mario_icons/lava.png',
    'sea': f'{cd.PROJECT_DIR}/data/mario_icons/sea.png',
    'sand': f'{cd.PROJECT_DIR}/data/mario_icons/sand.png',
    'grass': f'{cd.PROJECT_DIR}/data/mario_icons/grass.png',
    'chessboard': f'{cd.PROJECT_DIR}/data/mario_icons/chessboard.png',
    'chessboard_blue': f'{cd.PROJECT_DIR}/data/mario_icons/chessboard_blue.png',
    'chessboard_pink': f'{cd.PROJECT_DIR}/data/mario_icons/chessboard_pink.png',
    'brindle': f'{cd.PROJECT_DIR}/data/mario_icons/brindle.png',

    'mario': f'{cd.PROJECT_DIR}/data/mario_icons/mario.png',
    'luigi': f'{cd.PROJECT_DIR}/data/mario_icons/luigi.png',
    'peach': f'{cd.PROJECT_DIR}/data/mario_icons/peach.png',
    'bomb': f'{cd.PROJECT_DIR}/data/mario_icons/bomb.png',
    'goomba': f'{cd.PROJECT_DIR}/data/mario_icons/goomba.png',

    'green_mushroom': f'{cd.PROJECT_DIR}/data/mario_icons/green_mushroom.png',
    'star': f'{cd.PROJECT_DIR}/data/mario_icons/star.png',
    'red_mushroom': f'{cd.PROJECT_DIR}/data/mario_icons/red_mushroom.png',
    'coin': f'{cd.PROJECT_DIR}/data/mario_icons/coin.png',
    'cloud': f'{cd.PROJECT_DIR}/data/mario_icons/cloud.png'
}


def resize_with_transparency(img, size):
    pal = img.getpalette()
    width, height = img.size
    actual_transp = img.info['actual_transparency']  # XXX This will fail.

    result = Image.new('LA', img.size)

    im = img.load()
    res = result.load()
    for x in range(width):
        for y in range(height):
            t = actual_transp[im[x, y]]
            color = pal[im[x, y]]
            res[x, y] = (color, t)

    return result.resize(size, Image.ANTIALIAS)


def PNG_ResizeKeepTransparency(img, new_width=0, new_height=0, resample="LANCZOS", RefFile=''):
    # needs PIL
    # Inputs:
    #   - SourceFile  = initial PNG file (including the path)
    #   - ResizedFile = resized PNG file (including the path)
    #   - new_width   = resized width in pixels; if you need % plz include it here: [your%] *initial width
    #   - new_height  = resized hight in pixels ; default = 0 = it will be calculated using new_width
    #   - resample = "NEAREST", "BILINEAR", "BICUBIC" and "ANTIALIAS"; default = "ANTIALIAS"
    #   - RefFile  = reference file to get the size for resize; default = ''

    img = img.convert("RGBA")  # convert to RGBA channels
    width, height = img.size  # get initial size

    # if there is a reference file to get the new size
    if RefFile != '':
        imgRef = Image.open(RefFile)
        new_width, new_height = imgRef.size
    else:
        # if we use only the new_width to resize in proportion the new_height
        # if you want % of resize please use it into new_width (?% * initial width)
        if new_height == 0:
            new_height = new_width * width / height

    # split image by channels (bands) and resize by channels
    img.load()
    bands = img.split()
    # resample mode
    if resample == "NEAREST":
        resample = Image.NEAREST
    else:
        if resample == "BILINEAR":
            resample = Image.BILINEAR
        else:
            if resample == "BICUBIC":
                resample = Image.BICUBIC
            else:
                if resample == "ANTIALIAS":
                    resample = Image.ANTIALIAS
                else:
                    if resample == "LANCZOS":
                        resample = Image.LANCZOS
    bands = [b.resize((new_width, new_height), resample) for b in bands]
    # merge the channels after individual resize
    img = Image.merge('RGBA', bands)

    return img


def draw_mario_world(X, Y, agent_x, agent_y, target_x, target_y, agent_icon='goomba', target_icon='green_mushroom',
                     background_tile='lava', frame_tile='glass'):
    """
    This method creates the specified Mario's world.
    """
    # Initialize canvas
    W, H = 20, 20
    image = Image.new("RGBA", ((X + 2) * W, (Y + 2) * H), (255, 255, 255))
    # Define y offset for PIL
    agent_y = - (agent_y - (Y - 1))
    target_y = - (target_y - (Y - 1))
    # Set off-set due to frame_dict
    agent_x, agent_y = agent_x + 1, agent_y + 1
    target_x, target_y = target_x + 1, target_y + 1
    # Scale position to tile dimension
    agent_x, agent_y = agent_x * W, agent_y * H
    target_x, target_y = target_x * W, target_y * H
    # Load mario_icons and tiles
    agent_icon = Image.open(ICONS[agent_icon])
    target_icon = Image.open(ICONS[target_icon])
    background_tile = Image.open(ICONS[background_tile])
    frame_tile = Image.open(ICONS[frame_tile])
    # Resize mario_icons and tiles to fit the image

    background_tile = background_tile.resize((W, H), Image.LANCZOS)
    frame_tile = frame_tile.resize((W, H), Image.LANCZOS)
    agent_icon = PNG_ResizeKeepTransparency(agent_icon, new_width=int(W / 2), new_height=int(H / 2), resample="LANCZOS",
                                            RefFile='')
    target_icon = PNG_ResizeKeepTransparency(target_icon, new_width=int(W / 2) + 2, new_height=int(H / 2) + 2,
                                             resample="LANCZOS",
                                             RefFile='')
    # Define frame_dict tiles left corners
    frame_tiles_pos = []
    for i in range(Y + 2):
        frame_tiles_pos.append((0, i * H))
        frame_tiles_pos.append((4 * W, i * H))
        frame_tiles_pos.append((i * W, 0))
        frame_tiles_pos.append((i * W, 4 * H))
    # Define background_dict tiles left corners
    bkg_tiles_pos = []
    for i in range(1, Y + 1):
        bkg_tiles_pos.append((1 * W, i * H))
        bkg_tiles_pos.append((2 * W, i * H))
        bkg_tiles_pos.append((3 * W, i * H))
    # Draw frame_dict
    for box in frame_tiles_pos:
        image.paste(frame_tile, box=box)
    # Draw background_dict
    for box in bkg_tiles_pos:
        image.paste(background_tile, box=box)
    # Draw target_dict
    target_box = (target_x + 4, target_y + 4)
    image.paste(target_icon, box=target_box, mask=target_icon)
    # Draw agent_dict
    agent_box = (agent_x + 5, agent_y + 5)
    image.paste(agent_icon, box=agent_box, mask=agent_icon)

    return np.array(image)[:, :, :3]


def define_program(traj):
    """
    Translate the given trajectory in Mario program

    traj: list containing pairs of sequentially 2D Mario coordinates [((x0,y0),(x1,y1)), ((x1,y1),(x2,y2)),...]
    """
    program = []
    # Tras
    for (x0, y0), (x1, y1) in traj:
        if x0 < x1:
            program.append("right")
        elif x0 > x1:
            program.append("left")
        elif y0 < y1:
            program.append("up")
        elif y0 > y1:
            program.append("down")

    return program


def create_mario_dataset(folder):

    # List of 24 pairs of agent positions in a 3x3 grid
    position_set = set()
    for x, y in product([0, 1, 2], [0, 1, 2]):
        # Case 0,0
        if (x, y) == (0, 0):
            new_pos = ((0, 1), (1, 0))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 0,1
        elif (x, y) == (0, 1):
            new_pos = ((0, 0), (1, 1), (0, 2))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 0,2
        elif (x, y) == (0, 2):
            new_pos = ((0, 1), (1, 2))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 1,0
        elif (x, y) == (1, 0):
            new_pos = ((0, 0), (1, 1), (2, 0))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 1,1
        elif (x, y) == (1, 1):
            new_pos = ((0, 1), (1, 2), (2, 1), (1, 0))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 1,2
        elif (x, y) == (1, 2):
            new_pos = ((1, 1), (2, 2), (0, 2))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 2,0
        elif (x, y) == (2, 0):
            new_pos = ((1, 0), (2, 1))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 2,1
        elif (x, y) == (2, 1):
            new_pos = ((1, 1), (2, 0), (2, 2))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))
        # Case 2,2
        elif (x, y) == (2, 2):
            new_pos = ((2, 1), (1, 2))
            for n_p in new_pos:
                position_set.add(((x, y), n_p))

    # Order positions list for reproducibility
    position_list = list(position_set)
    position_list.sort(key=lambda y: (y[0], y[1]))
    # For each pair of positions, associate the move
    positions_move = []
    for pos1, pos2 in position_list:
        move = define_program([(pos1, pos2)])[0]
        positions_move.append((pos1, pos2, move))

    # Create Dataset from positions and moves
    folder = f'{cd.PROJECT_DIR}/data/mario/3x3'
    if not os.path.exists(folder):
        os.makedirs(folder)

    #agents = ['peach', 'mario', 'luigi', 'bomb', 'goomba']
    agents = ['mario']
    targets = ['star', 'coin', 'cloud', 'red_mushroom', 'green_mushroom']
    backgrounds = ['chessboard_blue', 'sea', 'grass', 'chessboard_pink', 'chessboard', 'sand','flowers1','flowers2']
    frames = ['brick', 'brick2', 'brindle', 'brick3', 'glass','concrete','wood']
    images = {'train': [],
              'val': [],
              'test': []}
    moves = {'train': [],
             'val': [],
             'test': []}
    positions = {'train': [],
                 'val': [],
                 'test': []}
    agent_dict = {'train': [],
                  'val': [],
                  'test': []}

    target_dict = {'train': [],
                   'val': [],
                   'test': []}

    background_dict = {'train': [],
                       'val': [],
                       'test': []}

    frame_dict = {'train': [],
                  'val': [],
                  'test': []}
    info = {
        'agent_dict': {'train': {c: 0 for c in agents},
                       'val': {c: 0 for c in agents},
                       'test': {c: 0 for c in agents}},
        'target_dict': {'train': {o: 0 for o in targets},
                        'val': {o: 0 for o in targets},
                        'test': {o: 0 for o in targets}},
        'background_dict': {'train': {c: 0 for c in backgrounds},
                            'val': {c: 0 for c in backgrounds},
                            'test': {c: 0 for c in backgrounds}},
        'frame_dict': {'train': {o: 0 for o in frames},
                       'val': {o: 0 for o in frames},
                       'test': {o: 0 for o in frames}},

        'pos': {'train': {str((p1, p2)): 0 for p1, p2, _ in positions_move},
                'val': {str((p1, p2)): 0 for p1, p2, _ in positions_move},
                'test': {str((p1, p2)): 0 for p1, p2, _ in positions_move}},

        'moves': {'train': {m: 0 for _, _, m in positions_move},
                  'val': {m: 0 for _, _, m in positions_move},
                  'test': {m: 0 for _, _, m in positions_move}}}

    configs = list(product(agents, targets, backgrounds, frames))
    # Order config list for reproducibility
    configs.sort()
    idxs_split = {'train': [],
                  'val': [],
                  'test': []}
    tot_imgs = len(configs) * len(positions_move)
    train_size = int(tot_imgs * 0.70)
    val_size = tot_imgs - train_size
    test_size = int(train_size * 0.10)
    train_size = train_size - test_size
    idxs = list(range(tot_imgs))
    random.seed(88888)
    idxs_split['train'] = random.choice(idxs, size=train_size, replace=False)
    idxs = [idx for idx in idxs if idx not in idxs_split['train']]
    idxs_split['val'] = random.choice(idxs, size=val_size, replace=False)
    idxs = [idx for idx in idxs if idx not in idxs_split['train'] and idx not in idxs_split['val']]
    idxs_split['test'] = random.choice(idxs, size=test_size, replace=False)

    idx = 0
    for config in tqdm(configs):
        (a, t, bg, f) = config
        for pos1, pos2, move in positions_move:
            img1 = draw_mario_world(X=3, Y=3, agent_x=pos1[0], agent_y=pos1[1], target_x=2, target_y=2, agent_icon=a,
                                    target_icon=t,
                                    background_tile=bg, frame_tile=f)
            img2 = draw_mario_world(X=3, Y=3, agent_x=pos2[0], agent_y=pos2[1], target_x=2, target_y=2, agent_icon=a,
                                    target_icon=t,
                                    background_tile=bg, frame_tile=f)

            for dataset in ['train', 'val', 'test']:
                if idx in idxs_split[dataset]:
                    images[dataset].append([img1, img2])
                    moves[dataset].append(move)
                    positions[dataset].append([pos1, pos2])
                    agent_dict[dataset].append(a)
                    target_dict[dataset].append(t)
                    background_dict[dataset].append(bg)
                    frame_dict[dataset].append(f)
                    info['target_dict'][dataset][t] += 1
                    info['frame_dict'][dataset][f] += 1
                    info['background_dict'][dataset][bg] += 1
                    info['agent_dict'][dataset][a] += 1
                    info['pos'][dataset][str((pos1, pos2))] += 1
                    info['moves'][dataset][move] += 1
            idx += 1
    # Check dimensions
    assert len(images['train']) == train_size
    assert len(images['val']) == val_size
    assert len(images['test']) == test_size

    # Save images, moves and positions
    for dataset in ['train', 'val', 'test']:
        np.save(os.path.join(folder, f'{dataset}_images.npy'), images[dataset])
        np.save(os.path.join(folder, f'{dataset}_moves.npy'), moves[dataset])
        np.save(os.path.join(folder, f'{dataset}_pos.npy'), positions[dataset])
        np.save(os.path.join(folder, f'{dataset}_agents.npy'), agent_dict[dataset])
        np.save(os.path.join(folder, f'{dataset}_targets.npy'), target_dict[dataset])
        np.save(os.path.join(folder, f'{dataset}_bkgs.npy'), background_dict[dataset])
        np.save(os.path.join(folder, f'{dataset}_frames.npy'), frame_dict[dataset])

    # Save info files
    with open(os.path.join(folder, 'info.json'), 'w') as file:
        json.dump(info, file, indent=4)
