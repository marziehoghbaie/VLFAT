import logging
import math
import os

import matplotlib.pylab as plt
import torch
import torchvision.transforms as T
from matplotlib.pyplot import figure


def transform_custom(images, image_size, model_type, gray_scale=True, resize=True):
    imgs = []

    for idx, image in enumerate(images):
        image = T.ToPILImage()(image)
        if gray_scale:
            image = T.Grayscale()(image)
        if resize:
            image = T.Resize((image_size, image_size))(image)
        image = T.ToTensor()(image)
        if gray_scale:
            image = T.Normalize(mean=[0.5], std=[0.5])(image)
        else:
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        imgs.append(image)

    if 'FSAViViT' == model_type:  # shape of model input: batch_size c t h w
        imgs = torch.stack(imgs)
        imgs = torch.transpose(imgs, dim0=0, dim1=1)
    elif 'FEViViT' == model_type or 'ResNet' == model_type or 'ViTTD_CNV1D' == model_type or \
            'ViT_VaR' == model_type or 'ViT_SinCos' == model_type or \
            'ViT_baseline' == model_type:  # shape of model input: batch_size t c h w
        imgs = torch.stack(imgs)

    return imgs


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


class CheckpointSaver:
    def __init__(self, save_dir, checkpoint_fn='ckpt.pth.tar', best_fn='ckpt.best.pth.tar',
                 best_step_fn='ckpt.best.step{}.pth.tar', save_best_step=False, lr_steps=[]):
        """
        Only mandatory: save_dir
            Can configure naming of checkpoints files through checkpoint_fn, best_fn and best_stage_fn
            If you want to keep best-performing checkpoints per step
        """

        self.save_dir = save_dir

        # checkpoints names
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn

        # save best per step?
        self.save_best_step = save_best_step
        self.lr_steps = []

        # init var to keep track of best performing checkpoints
        self.current_best = 0

        # save best at each step?
        if self.save_best_step:
            assert lr_steps != [], "Since save_best_step=True, need proper value for lr_steps. Current: {}".format(
                lr_steps)
            self.best_for_stage = [0] * (len(lr_steps) + 1)

    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoints and keeps copy if current perf is the best overall or [optional] best for current LR step
        """

        # save last checkpoints
        self.checkpoint_fn = 'current_val_acc_{}_ckpt.pth.tar'.format(current_perf)
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)

        # keep track of best model
        self.is_best = current_perf > self.current_best
        if self.is_best:
            self.current_best = current_perf
            self.best_fn = 'best_val_acc_{}_ckpt.pth.tar'.format(current_perf)
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # keep track of best-performing model per step [optional]
        if self.save_best_step:

            assert epoch >= 0, "Since save_best_step=True, need proper value for 'epoch'. Current: {}".format(epoch)
            s_idx = sum(epoch >= l for l in self.lr_steps)
            self.is_best_for_stage = current_perf > self.best_for_stage[s_idx]

            if self.is_best_for_stage:
                self.best_for_stage[s_idx] = current_perf
                best_stage_fp = os.path.join(self.save_dir, self.best_stage_fn.format(s_idx))
            save_dict['best_prec_per_stage'] = self.best_for_stage

        # save
        torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        # if self.is_best:
        #     shutil.copyfile(checkpoint_fp, best_fp)
        # if self.save_best_step and self.is_best_for_stage:
        #     shutil.copyfile(checkpoint_fp, best_stage_fp)

    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage', None)


def load_model(load_path, model, optimizer=None, allow_size_mismatch=False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    # -- load dictionary
    assert os.path.isfile(load_path), "Error when loading the model, provided path not found: {}".format(load_path)
    if not torch.cuda.is_available():
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(load_path)

    if 'model_state_dict' in checkpoint.keys():
        loaded_state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint.keys():
        loaded_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint.keys():
        loaded_state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('[INTERNAL ERROR] the checkpoint does not contain any model object ...')

    if allow_size_mismatch:
        loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
        model_state_dict = model.state_dict()
        model_sizes = {k: v.shape for k, v in model_state_dict.items()}
        mismatched_params = []
        for k in loaded_sizes:
            if loaded_sizes[k] != model_sizes[k]:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]

    # -- copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch_idx'], checkpoint
    return model


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori * reduction_ratio)


class InverseSquareRootScheduler:
    def __init__(self, warmup_init_lr, warmup_end_lr, warmup_updates=4000):
        """Decay the LR based on the inverse square root of the update number.
        We also support a warmup phase where we linearly increase the learning rate
        from some initial learning rate (``--warmup-init-lr``) until the configured
        learning rate (``--lr``). Thereafter we decay proportional to the number of
        updates, with a decay factor set to align with the configured learning rate.
        During warmup::
          lrs = torch.linspace(self.lr_ori, self.warmup_end_lr, self.warmup_updates)
          lr = lrs[update_num]
        After warmup::
          decay_factor = cfg.lr * sqrt(cfg.warmup_updates)
          lr = decay_factor / sqrt(update_num)
        """
        self.warmup_init_lr = warmup_init_lr
        """warmup the learning rate linearly for the first N updates"""
        self.warmup_updates = warmup_updates
        self.warmup_end_lr = warmup_end_lr
        # linearly warmup for the first cfg.warmup_updates
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / self.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = self.warmup_end_lr * self.warmup_updates ** 0.5

        # initial learning rate
        self.lr = self.warmup_init_lr

    def adjust_lr(self, optimizer, num_updates):
        # print('[INFO] Number of current iterations is {}'.format(num_updates))
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5

        change_lr_on_optimizer(optimizer, self.lr)


def draw_results(results, save_path=None):
    assert save_path is not None

    epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val = results

    figure(figsize=(8, 6))
    plt.plot(epochs_loss)
    plt.plot(epochs_loss_val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path + '/' + 'loss.png')

    figure(figsize=(8, 6))
    plt.plot(epochs_acc)
    plt.plot(epochs_acc_val)
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path + '/' + 'acc.png')


def create_logger(save_path, name=''):
    filename = save_path + '/' + 'log_{}.txt'.format(name)
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a+')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger
