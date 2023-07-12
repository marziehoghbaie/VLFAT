# for file level performance only
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

import torch
from test_files.test import test_complete_Robustness
from config.load_config import read_conf_file
from utils import utils, model_utils, data_utils
import numpy as np

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


def main(config_path):
    model_config, dataset_info, train_config, log_info, where, config, \
        device, model_layout, check_point_path, model_name, results_path, logger = read_conf_file(config_path)
    model = model_utils.ViT_create_model(model_config, model_layout, logger)
    config_name = config_path.split(sep='/')[-1].split(sep='.')[0]
    if not train_config['train']:  # if you're loading weights in pretraining mode or
        # loading in test mode
        model = utils.load_model(check_point_path, model, allow_size_mismatch=train_config['allow_size_mismatch'])
        logger.info('[INFO] The checkpoint is loaded from {} ...'.format(check_point_path))

    logger.info('[INFO] training config ... \n {} \n\n'.format(config))
    model = model.to(device)
    assert not train_config['allow_size_mismatch'] and not train_config['train']
    if dataset_info['dataset_name'] == 'OLIVES':
        volume_resolutions = [5, 15, 25, 35, 45]
    else:
        volume_resolutions = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]
    logger.info('[INFO] List of frames that are tested: {}'.format(volume_resolutions))
    for num_frame_test in volume_resolutions:
        model_config['num_frames'] = num_frame_test
        dataset_info['loader_type'] = 'random_middle'
        test_loader, _ = data_utils.create_dataloaders(dataset_info,
                                                       train_config,
                                                       model_config,
                                                       model_layout,
                                                       logger,
                                                       where)
        # to assure that required changes are applied
        assert test_loader.dataset.loader_type == 'random_middle' and test_loader.dataset.n_frames == num_frame_test
        model = model.to(device)
        balanced_accs = []
        num_tests = model_config['num_test']
        num_frames = model_config['num_frames']

        for test_iter in (range(0, num_tests)):
            balanced_acc = test_complete_Robustness(test_loader, model,
                                                    logger, phase='test',
                                                    save_path=results_path, device=device,
                                                    n_test=test_iter)
            balanced_accs.append(balanced_acc)

        balanced_acc_std = np.std(np.array(balanced_accs))
        balanced_accs = np.average(np.array(balanced_accs))

        logger.info(f' Model name: {config_name},'
                    f' Results over {num_tests} iterations for #B-scans {num_frames}:\n '
                    f' Balanced Average Accuracy: {balanced_accs}'
                    f' std of balanced acc: {balanced_acc_std}.')
        logger.info('_________________________________________________________________________________________________')
    # test the whole volume
    test_loader, _ = data_utils.create_dataloaders(dataset_info,
                                                   train_config,
                                                   model_config,
                                                   model_layout,
                                                   logger,
                                                   where)
    test_loader.dataset.loader_type = 'variable'
    # to assure that required changes are applied
    assert test_loader.dataset.loader_type == 'variable'
    model = model.to(device)
    balanced_acc = test_complete_Robustness(test_loader, model,
                                            logger, phase='test',
                                            save_path=results_path, device=device,
                                            n_test=0)

    logger.info(f'Model name: {config_name}, Balanced Accuracy for whole volumes: {balanced_acc}.')
    logger.info('___________________________________________________________________________________________________')
    logger.info(f'~~~~~~~~~~~~~~~~~~~~~~~~Model name: {config_name} test is finished!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.info('___________________________________________________________________________________________________')
