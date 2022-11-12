import os
import torch
import json
import glob as gb

from datetime import datetime
from data.dataloader import FlowLoader



def setup_path(args):
    dataset = args.dataset
    batch_size = args.batch_size
    resolution = args.resolution
    inference = args.inference
    frames = args.frames
    gaps = args.gaps
    queries = args.queries

    # Make all the essential folders, e.g. models, logs, results, etc.
    global dt_string, logPath, modelPath, resultsPath
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    
    """ TODO: fill in the log & ckpt paths """
    log_dir = '/path/to/logdir/'

    os.makedirs(log_dir + '/logs/', exist_ok=True)
    os.makedirs(log_dir + '/models/', exist_ok=True)


    logPath = os.path.join(log_dir + '/logs/', f'{dt_string}-dataset_{dataset}-'
                                      f'OCLR-'
                                      f'res_{resolution[0]}x{resolution[1]}-t_{frames}-gap_{gaps}-'
                                      f'bs_{batch_size}-queries_{queries}')

    modelPath = os.path.join(log_dir + '/models/', f'{dt_string}-dataset_{dataset}-'
                                      f'OCLR-'
                                      f'res_{resolution[0]}x{resolution[1]}-t_{frames}-gap_{gaps}-'
                                      f'bs_{batch_size}-queries_{queries}')
   
    if not inference:
        os.makedirs(logPath, exist_ok=True)
        os.makedirs(modelPath, exist_ok=True)

        # Save all the experiment settings.
        with open('{}/running_command.txt'.format(modelPath), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return [logPath, modelPath]


def setup_dataset(args):
    resolution = args.resolution  # h,w
    dataset = args.dataset
    frames = args.frames
    gaps = args.gaps
    gapsets = [int(x) for x in gaps.split(',')]
    in_channels = int(len(gapsets) * 2)
    out_channels = args.queries
    amodal = False
    
    """ TODO: fill in dataset paths """
     
    if dataset == 'Syn':
        trn_basepath = '/path/to/Syn-train/'
        trn_img_dir = trn_basepath + '/JPEGImages'
        trn_gt_dir = trn_basepath + '/Annotations'
        trn_seq = [os.path.basename(x) for x in gb.glob(os.path.join(trn_img_dir, '*'))]

        val_basepath = '/path/to/Syn-val/'
        val_img_dir = val_basepath + '/JPEGImages'
        val_gt_dir = val_basepath + '/Annotations'
        val_seq = [os.path.basename(x) for x in gb.glob(os.path.join(val_img_dir, '*'))]
        
        trn_flow_dir = trn_basepath + '/Flows_gap1'
        val_flow_dir = val_basepath + '/Flows_gap1'
        trn_data_dir = [trn_flow_dir, trn_img_dir, trn_gt_dir]
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        
        gt_res = resolution  
        amodal = True  
       
   
    elif dataset == 'DAVIS16':      
        trn_basepath = '/path/to/DAVIS/'
        trn_img_dir = trn_basepath + '/JPEGImages/480p'
        trn_gt_dir = trn_basepath + '/Annotations/480p'
        trn_flow_dir = trn_basepath + '/Flows_gap1'
        trn_seq =  ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 
                    'dog-agility', 'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 
                    'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 
                    'paragliding', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 
                    'swing', 'tennis', 'train']
        trn_data_dir = [trn_flow_dir, trn_img_dir, trn_gt_dir]

        val_basepath = '/path/to/DAVIS/'
        val_img_dir = val_basepath + '/JPEGImages/480p'
        val_gt_dir = val_basepath + '/Annotations/480p'
        val_flow_dir = val_basepath + '/Flows_gap1'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']            
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]

        gt_res = None

            
    elif dataset == 'Segtrack':

        val_basepath = '/path/to/SegTrackv2/'
        val_img_dir = val_basepath + '/JPEGImages'
        val_gt_dir = val_basepath + '/Annotations'
        val_flow_dir = val_basepath + '/Flows_gap1'
        val_seq = ['drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
                    'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise']   
                
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        trn_basepath = '/scratch/shared/beegfs/jyx/SegTrackv2'
        trn_seq = None
        trn_data_dir = val_data_dir

        gt_res = None
       
   
        
    elif dataset == 'FBMS': 
        val_basepath = '/path/to/FBMS-59/'
        val_img_dir = val_basepath + '/JPEGImages'
        val_gt_dir = val_basepath + '/Annotations'
        val_flow_dir = val_basepath + '/Flows_gap1'
        val_seq = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
                    'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
                    'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
                    'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
        
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        trn_basepath = '/path/to/FBMS-59/'
        trn_seq = ['bear01', 'bear02', 'cars2', 'cars3', 'cars6', 'cars7', 'cars8', 'cars9', 'cats02', 'cats04', 
                   'cats05', 'cats07', 'ducks01', 'horses01', 'horses03', 'horses06', 'lion02', 'marple1', 'marple10',
                   'marple11', 'marple13', 'marple3', 'marple5', 'marple8', 'meerkats01', 'people04', 'people05',
                   'rabbits01', 'rabbits05']
        trn_data_dir = val_data_dir
        gt_res = None
     
    elif dataset == 'MoCA':
        trn_basepath = '/path/to/MoCA_filter/'
        trn_img_dir = trn_basepath + '/JPEGImages'
        trn_gt_dir = trn_basepath + '/Annotations'

        trn_seq =  None
        
        val_basepath = '/path/to/MoCA_filter/'
        val_img_dir = val_basepath + '/JPEGImages'
        val_gt_dir = val_basepath + '/Annotations'
        val_seq = ['arabian_horn_viper', 'arctic_fox_1', 'arctic_wolf_1', 'black_cat_1', 'crab', 'crab_1', 
                    'cuttlefish_0', 'cuttlefish_1', 'cuttlefish_4', 'cuttlefish_5', 
                    'devil_scorpionfish', 'devil_scorpionfish_1', 'flatfish_2', 'flatfish_4', 'flounder', 
                    'flounder_3', 'flounder_4', 'flounder_5', 'flounder_6', 'flounder_7', 
                    'flounder_8', 'flounder_9', 'goat_1', 'hedgehog_1', 'hedgehog_2', 'hedgehog_3', 
                    'hermit_crab', 'jerboa', 'jerboa_1', 'lion_cub_0', 'lioness', 'marine_iguana', 
                    'markhor', 'meerkat', 'mountain_goat', 'peacock_flounder_0', 
                    'peacock_flounder_1', 'peacock_flounder_2', 'polar_bear_0', 'polar_bear_2', 
                    'scorpionfish_4', 'scorpionfish_5', 'seal_1', 'shrimp', 
                    'snow_leopard_0', 'snow_leopard_1', 'snow_leopard_2', 'snow_leopard_3', 'snow_leopard_6', 
                    'snow_leopard_7', 'snow_leopard_8', 'spider_tailed_horned_viper_0', 
                    'spider_tailed_horned_viper_2', 'spider_tailed_horned_viper_3',
                    'arctic_fox', 'arctic_wolf_0', 'devil_scorpionfish_2', 'elephant', 
                    'goat_0', 'hedgehog_0', 
                    'lichen_katydid', 'lion_cub_3', 'octopus', 'octopus_1', 
                    'pygmy_seahorse_2', 'rodent_x', 'scorpionfish_0', 'scorpionfish_1', 
                    'scorpionfish_2', 'scorpionfish_3', 'seal_2',
                    'bear', 'black_cat_0', 'dead_leaf_butterfly_1', 'desert_fox', 'egyptian_nightjar', 
                    'pygmy_seahorse_4', 'seal_3', 'snowy_owl_0',
                    'flatfish_0', 'flatfish_1', 'fossa', 'groundhog', 'ibex', 'lion_cub_1', 'nile_monitor_1',
                    'polar_bear_1', 'spider_tailed_horned_viper_1']
        

        trn_flow_dir = trn_basepath + '/Flows_gap1'
        val_flow_dir = val_basepath + '/Flows_gap1'
        trn_data_dir = [trn_flow_dir, trn_img_dir, trn_gt_dir]
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        gt_res = None
       
    elif dataset == 'DAVIS17m':
        trn_basepath = '/path/to/DAVIS2017-motion/'

        trn_img_dir = trn_basepath + '/JPEGImages/480p'
        trn_gt_dir = trn_basepath + '/Annotations/480p'
        trn_seq =  ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 
                   'bus', 'car-turn', 'cat-girl', 'classic-car', 'crossing',
                   'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'drift-turn', 'elephant', 'flamingo', 'hike',
                   'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
                   'lady-running', 'longboard', 'lucia', 'mallard-fly',
                   'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike',
                   'night-race', 'paragliding', 'rallye', 'rhino',
                   'rollerblade', 'scooter-board', 'scooter-gray', 'skate-park', 'snowboard', 'soccerball', 'stroller',
                   'stunt', 'surf', 'swing', 'tennis', 'tractor-sand', 'varanus-cage', 'walking']
                   #color-run, dog-gooses, dogs-scale, drone, lindy-hop, schoolgirls, sheep, train, tuk-tuk, upside-down, planes-water


        val_basepath = '/path/to/DAVIS2017-motion/'
        val_img_dir = val_basepath + '/JPEGImages/480p'
        val_gt_dir = val_basepath + '/Annotations/480p'
        
        val_seq = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow',
                  'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish',
                     'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
                    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
        
        trn_flow_dir = trn_basepath + '/Flows_gap1'
        val_flow_dir = val_basepath + '/Flows_gap1'
        trn_data_dir = [trn_flow_dir, trn_img_dir, trn_gt_dir]
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        print(val_data_dir)

        gt_res = None
    
    else:
        raise ValueError('Unknown Setting.')


    if args.inference:
        trn_dataset = None
        print("inference")
    else:
        trn_dataset = FlowLoader(data_dir=trn_data_dir, resolution=resolution, pair_list=gapsets, 
                                   data_seq=trn_seq, dataset = dataset, train = True, out_channels = out_channels, gt_res = gt_res, frames = frames, amodal = amodal)

    val_dataset = FlowLoader(val_data_dir, resolution, gapsets, val_seq, dataset, False, out_channels, gt_res, frames, amodal)
    
    return trn_dataset, val_dataset, resolution, in_channels, out_channels


    
    
