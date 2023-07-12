# for file level performance only
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

import argparse
from config.args_parser import get_args_parser

import torch
import torch.nn as nn

from train_files.train import train_val
from test_files.test import test_complete
from config.load_config import read_conf_file
from utils import utils, model_utils, optimizer_utils, data_utils, scheduler_utils

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


def main(config_path):
    model_config, dataset_info, train_config, log_info, where, config, \
        device, model_layout, check_point_path, model_name, results_path, logger = read_conf_file(config_path)

    model = model_utils.ViT_create_model(model_config, model_layout, logger)
    VLFAT = False
    if model_config['model_type'] == 'ViT_VaR':
        VLFAT = True

    init_epoch = 0
    """ Setting up the loss function, optimizer, and schedulers"""
    """Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer"""
    # set the loss function and the optimizers
    cycle_momentum = True
    if train_config['optimizer'] == 'adam':
        cycle_momentum = False
    optimizer = optimizer_utils.create_optimizer(model, train_config)

    if train_config['train']:
        train_loader, len_train_set, val_loader, len_val_set = data_utils.create_dataloaders(dataset_info,
                                                                                             train_config,
                                                                                             model_config,
                                                                                             model_layout,
                                                                                             logger,
                                                                                             where)

        if model_config['weighted']:
            weight_type = 'balanced'
        else:
            weight_type = None

        classes_weights = train_loader.dataset.cal_cls_weight(weight_type=weight_type)
        loss_fn = nn.CrossEntropyLoss(weight=classes_weights).to(device)

        scheduler = scheduler_utils.create_scheduler(optimizer,
                                                     train_config,
                                                     cycle_momentum,
                                                     logger,
                                                     len_train_set)

    ckpt_saver = utils.CheckpointSaver(results_path)
    model = model.to(device)
    logger.info(model)
    if train_config['resume'] and train_config['train']:  # you resume the training procedure for the exact same model
        model, optimizer, epoch_idx, ckpt_dict = utils.load_model(check_point_path,
                                                                  model,
                                                                  optimizer,
                                                                  allow_size_mismatch=train_config[
                                                                      'allow_size_mismatch'])
        init_epoch = epoch_idx
        ckpt_saver.set_best_from_ckpt(ckpt_dict)
        logger.info('[INFO] The training is going to resume from {} ...'.format(check_point_path))
    elif train_config['pretrain'] or not train_config['train']:  # if you're loading weights in pretraining mode or
        # loading in test mode
        model = utils.load_model(check_point_path, model, allow_size_mismatch=train_config['allow_size_mismatch'])
        logger.info('[INFO] The checkpoint is loaded from {} ...'.format(check_point_path))

    logger.info('[INFO] training config ... \n {} \n\n'.format(config))

    model = model.to(device)
    if train_config['train']:
        train_val(train_config, logger, model, optimizer, loss_fn, train_loader, val_loader, scheduler, ckpt_saver,
                  init_epoch, device, VLFAT)
    else:
        assert not train_config['allow_size_mismatch'] and not train_config['train']
        test_loader, len_test_set = data_utils.create_dataloaders(dataset_info,
                                                                  train_config,
                                                                  model_config,
                                                                  model_layout,
                                                                  logger,
                                                                  where)
        loss_fn = nn.CrossEntropyLoss().to(device)
        test_complete(test_loader, model, loss_fn, logger, phase='test', save_path=results_path, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT3D', parents=[get_args_parser()])
    args = parser.parse_args()
    config_path = args.config_path
    main(config_path)
