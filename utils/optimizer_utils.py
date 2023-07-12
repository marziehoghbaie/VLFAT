import torch
import torch.optim as optim
from adan_pytorch import Adan


# ToDo Add SAM and Ranger optimizer
def create_optimizer(model, train_config):
    if train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_config['init_lr'],
                                    momentum=train_config['momentum'],
                                    weight_decay=train_config['weight_decay']
                                    )  # add weight_decay = 0.01
    elif train_config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=train_config['init_lr'],
                               weight_decay=train_config['weight_decay'])
        cycle_momentum = False
    elif train_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=train_config['init_lr'],
                                      weight_decay=train_config['weight_decay'],
                                      # betas=(0.9, 0.999)  # applied on swin_video transformer
                                      )
    elif train_config['optimizer'] == 'adan':
        optimizer = Adan(
            model.parameters(),
            lr=train_config['init_lr'],  # learning rate
            betas=(0.1, 0.1, 0.001),  # beta 1-2-3 as described in paper
            weight_decay=train_config['weight_decay']  # weight decay
        )
    return optimizer
