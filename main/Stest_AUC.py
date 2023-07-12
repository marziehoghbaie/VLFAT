# for file level performance only
import inspect
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

from config.load_config import read_conf_file
from utils import utils, model_utils, data_utils
from test_files.plot_roc import binary_auc, multiclass_auc

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


def main(configs_list):
    trues, probs = [], []
    for color, model_type, config_path in configs_list:
        model_config, dataset_info, train_config, log_info, where, config, \
            device, model_layout, check_point_path, model_name, results_path, logger = read_conf_file(config_path)
        logger.info(f'color: {color}, model_type: {model_type}, config_path{config_path}')

        model = model_utils.ViT_create_model(model_config, model_layout, logger)

        assert dataset_info['dataset_name'] in ['Duke', 'OLIVES']

        if not train_config['train']:  # if you're loading weights in pretraining mode or
            # loading in test mode
            model = utils.load_model(check_point_path, model, allow_size_mismatch=train_config['allow_size_mismatch'])
            logger.info('[INFO] The checkpoint is loaded from {} ...'.format(check_point_path))

        logger.info('[INFO] training config ... \n {} \n\n'.format(config))
        model = model.to(device)

        assert not train_config['allow_size_mismatch'] and not train_config['train']
        logger.info('.................................................................................................')
        test_loader, _ = data_utils.create_dataloaders(dataset_info,
                                                       train_config,
                                                       model_config,
                                                       model_layout,
                                                       logger,
                                                       where)

        model = model.to(device)
        model.eval()
        y_true = []
        y_prob = []
        running_corrects = 0
        '//////////////////////////////////////////////////////////////////////////////////////////////////////////////'
        with torch.no_grad():
            for idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)  # accelerator handles it
                logits = model(images)
                _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
                prob = F.softmax(logits, dim=1).data
                y_true.extend(labels.cpu().detach().numpy())
                y_prob.append(prob.cpu().detach().numpy()[0])
                running_corrects += preds.eq(labels.view_as(preds)).sum().item()
        trues.append(np.array(y_true))
        probs.append(np.array(y_prob))
    trues = np.array(trues)
    probs = np.array(probs)

    if model_config['num_classes'] == 2:
        binary_auc(trues, probs, configs_list, results_path)
    else:
        multiclass_auc(trues, probs, configs_list, results_path)


if __name__ == '__main__':
    configs_list = [
        ("lightcoral", 'resnet_avg', 'config/YML_files/resnet_pool.yaml'),
        ("slateblue", 'FE ViT3D', 'config/YML_files/FE_ViViT.yaml'),
        ("red", 'FSA ViT3D', 'config/YML_files/FSA_ViViT.yaml'),
    ]
    main(configs_list)
