import torch
import torch.optim as optim

from utils.utils import CosineScheduler, InverseSquareRootScheduler
import model_zoo


def create_scheduler(optimizer, train_config, cycle_momentum, logger, len_train_set):
    steps = train_config['step_coeff'] * len_train_set
    scheduler = None

    if train_config['scheduler'] == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=train_config['base_lr_cyclic'],
                                                      max_lr=train_config['max_lr'],
                                                      step_size_up=steps,
                                                      mode='triangular2',
                                                      cycle_momentum=cycle_momentum)
    elif train_config['scheduler'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['num_epochs'])
    elif train_config['scheduler'] == 'cosine_scheduler':
        scheduler = CosineScheduler(train_config['init_lr'], train_config['num_epochs'])
    elif train_config['scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif train_config['scheduler'] == 'isr':
        scheduler = InverseSquareRootScheduler(warmup_init_lr=train_config['init_lr'],
                                               warmup_end_lr=train_config['max_lr'],
                                               warmup_updates=steps)
    elif train_config['scheduler'] == 'cosine_with_warmup':
        """
        optimizer (Optimizer) – Wrapped optimizer.
        T_0 (int) – Number of iterations for the first restart.
        T_mult (int, optional) – A factor increases Ti after a restart. Default: 1.
        eta_min (float, optional) – Minimum learning rate. Default: 0.
        last_epoch (int, optional) – The index of last epoch. Default: -1.
        verbose (bool) – If True, prints a message to stdout for each update. Default: False.
        """
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=train_config['warmup_epochs'],
                                                                   T_mult=train_config['T_mult'],
                                                                   eta_min=train_config['eta_min'],
                                                                   last_epoch=train_config['last_epoch'],
                                                                   verbose=False)

    return scheduler
