# for file level performance only
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

import time
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from test_files.test import test
from utils.utils import *


def train_val(train_config,
              logger,
              model,
              optimizer,
              loss_fn,
              train_loader,
              val_loader,
              scheduler,
              ckpt_saver,
              init_epoch=0,
              device='cpu',
              VLFAT=False):

    writer = SummaryWriter(log_dir=ckpt_saver.save_dir)
    best_acc = 0.0
    epochs_acc = []
    epochs_loss = []
    epochs_loss_val = []
    epochs_acc_val = []

    # train loop
    scaler = torch.cuda.amp.GradScaler()
    num_updates = 0  # applicable for InverseSquareRootScheduler
    model = model.to(device)
    for epoch in range(init_epoch, train_config['num_epochs']):
        model.train()

        # loop over dataset
        logger.info('[INFO] Current learning rate at epoch {}: {}'.format(epoch, showLR(optimizer)))
        print('[INFO] Train loop ...')
        running_corrects = 0
        running_loss = 0
        running_all = 0
        epoch_time = time.time()
        for idx, (images, labels) in (enumerate(train_loader)):
            optimizer.zero_grad()
            # Forward Pass
            images, labels = images.to(device), labels.to(device)
            images, labels = map(Variable, (images, labels))
            logits = model(images)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            current_lr = showLR(optimizer)

            # Compute Loss and Perform Back-propagation
            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)
            running_all += images.size(0)

            # Scale Gradients
            scaler.scale(loss).backward()
            # Update Optimizer
            scaler.step(optimizer)
            scaler.update()

            # Update the scheduler...
            if train_config['scheduler'] == 'cyclic':
                scheduler.step()
            elif train_config['scheduler'] == 'isr':
                scheduler.adjust_lr(optimizer, num_updates)
            elif train_config['scheduler'] == 'cosine_with_warmup':
                scheduler.step()
            num_updates += 1

            if 0 == (idx + 1) % 50:
                # tensorboard update
                writer.add_scalar('ACC/train/batch', running_corrects / running_all, idx)
                writer.add_scalar('Loss/train/batch', running_loss / running_all, idx)
                # logger update
                logger.info("batch id: {}, batch acc: {}, batch loss: {}".format(idx,
                                                                                 running_corrects / running_all,
                                                                                 running_loss / running_all))
                logger.info('[INFO] Current learning rate (get from optimizer): {}'.format(showLR(optimizer)))

        print('[INFO] Validation loop ...')
        logger.info('[INFO] This training epoch took {} sec'.format(time.time() - epoch_time))
        val_time = time.time()
        'accuracy, loss, balanced_acc'
        _, val_loss, val_acc = test(val_loader, model, loss_fn, logger, phase='val')

        logger.info('[INFO] This validation epoch took {} sec'.format(time.time() - val_time))

        if train_config['scheduler'] == 'cosine_annealing':
            scheduler.step()

        # -- save the best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_dict = {
                'epoch_idx': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "scaler": scaler.state_dict()
            }
            ckpt_saver.save(save_dict, val_acc)
            logger.info('[INFO] the checkpoint is saved at epoch {} and with val acc {}'.format(epoch, val_acc))

        epoch_acc = running_corrects / running_all
        epoch_loss = running_loss / running_all

        # Tensorboard setting
        # train
        writer.add_scalar('ACC/train', epoch_acc, epoch)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        # val
        writer.add_scalar('ACC/val', val_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        logger.info("epoch: {}, epoch acc: {}, epoch loss: {},"
                    " val acc: {}, val loss: {} with current lr {}".format(epoch,
                                                                           epoch_acc,
                                                                           epoch_loss,
                                                                           val_acc,
                                                                           val_loss,
                                                                           current_lr))

        # # # # # # # # # # # #

        epochs_acc.append(epoch_acc)
        epochs_loss.append(epoch_loss)

        epochs_acc_val.append(val_acc)
        epochs_loss_val.append(val_loss)

        if VLFAT:
            """part for variable length"""
            var_length = [5, 10, 15, 20, 25]
            n_b_scans_new = np.random.choice(var_length)
            train_loader.dataset.n_frames = n_b_scans_new
            val_loader.dataset.n_frames = n_b_scans_new
            logger.info('the number of b-scans is set to {}'.format(train_loader.dataset.n_frames))

    writer.close()
    draw_results([epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val], save_path=ckpt_saver.save_dir)
