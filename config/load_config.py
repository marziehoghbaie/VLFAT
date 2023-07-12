import inspect
import json
import os
import sys
import time

import torch
import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
sys.path.insert(0, grandparentdir)
from utils import utils


def read_conf_file(config_path):
    file = open(config_path, mode='r')
    config = yaml.safe_load(file)
    model_config = config[0]["model_inputs"]
    dataset_info = config[1]["dataset"]
    train_config = config[2]['train_config']
    log_info = config[3]['log']
    where = config[4]['where']

    device = torch.device('cuda' if torch.cuda.is_available() and train_config['use_gpu'] else 'cpu')

    model_layout = model_config[model_config['model_type']]  # model configurations
    check_point_path = where + '/' + train_config['load_path'] + '/' + train_config['checkpoint']

    #  prepare Results path
    model_name = '_'.join(
        [model_config['model_type'], str(model_config['image_size']), dataset_info['dataset_name']])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_path = where + '/' + log_info['save_path'] + '/' + model_name + '/' + timestr
    os.makedirs(results_path, exist_ok=True)
    logger = utils.create_logger(results_path)
    logger.info('[INFO] Device {}'.format(device))
    logger.info('[INFO] Results are saved in {} ...'.format(results_path))

    """ write current model configs into json files """
    with open(results_path + '/' + model_name + ".json", "w") as outfile:
        json.dump(model_config, outfile)
    with open(results_path + '/' + "config.json", "w") as outfile:
        json.dump(config, outfile)

    return model_config, dataset_info, train_config, log_info, where, config, \
           device, model_layout, check_point_path, model_name, results_path, logger

