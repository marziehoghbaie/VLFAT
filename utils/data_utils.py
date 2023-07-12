import random

from torch.utils.data import DataLoader

from data.channel_wise_aug import available_augmentations
from data.data_loader_multi_class import Multiclass_ds
from data.data_loader_binary import OLIVES_ds
from data.data_loader_duke import Duke_ds


def create_dataloaders(dataset_info, train_config, model_config, model_layout, logger, where):
    gray_scale = True if model_config['channels'] == 1 else False
    if not train_config['train']:
        if dataset_info['dataset_name'] == 'Multi_class':
            logger.info('[INFO] load test set from {}...'.format(dataset_info['annotation_path_test']))
            logger.info('[INFO] load test set ...')
            test_set = Multiclass_ds(loader_type=dataset_info['loader_type'],
                                     annotation_path=where + '/' + dataset_info['annotation_path_test'],
                                     image_size=model_config['image_size'],
                                     n_frames=model_config['num_frames'],
                                     categories=['cnv1', 'cnv2', 'cnv3',
                                                 'dme', 'ga', 'healthy',
                                                 'iamd', 'rvo', 'stargardt'],
                                     model_type=model_config['model_type'],
                                     gray_scale=gray_scale,
                                     logger=logger)
        elif dataset_info['dataset_name'] == 'OLIVES':
            logger.info('[INFO] load test set from {}...'.format(dataset_info['annotation_path_test']))
            logger.info('[INFO] load test set ...')
            test_set = OLIVES_ds(loader_type=dataset_info['loader_type'],
                                 annotation_path=where + '/' + dataset_info['annotation_path_test'],
                                 image_size=model_config['image_size'],
                                 n_frames=model_config['num_frames'],
                                 categories=['dr', 'dme'],
                                 model_type=model_config['model_type'],
                                 gray_scale=gray_scale,
                                 logger=logger)
        elif dataset_info['dataset_name'] == 'Duke':
            logger.info('[INFO] load test set from {}...'.format(dataset_info['annotation_path_test']))
            logger.info('[INFO] load test set ...')
            test_set = Duke_ds(loader_type=dataset_info['loader_type'],
                               annotation_path=where + '/' + dataset_info['annotation_path_test'],
                               image_size=model_config['image_size'],
                               categories=["AMD", "Normal"],
                               model_type=model_config['model_type'],
                               gray_scale=gray_scale, n_frames=model_config['num_frames'],
                               logger=logger, where=where)

        logger.info('[INFO] number of samples in test set: {}'.format(test_set.__len__()))
        # initialize the dataloader
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=train_config['batch_size'],
            shuffle=dataset_info['shuffle'],
            num_workers=dataset_info['num_workers'],
            pin_memory=False
        )
        return test_loader, test_set.__len__()
    else:
        """ load datasets (type of supported Datasets) """
        random.shuffle(available_augmentations)
        if dataset_info['dataset_name'] == 'Multi_class':
            logger.info('Load dataset for training ...')
            train_set = Multiclass_ds(loader_type=dataset_info['loader_type'],
                                      annotation_path=where + '/' + dataset_info['annotation_path_train'],
                                      augment=True,
                                      augmentation_list=available_augmentations,
                                      image_size=model_config['image_size'],
                                      categories=['cnv1', 'cnv2', 'cnv3',
                                                  'dme', 'ga', 'healthy',
                                                  'iamd', 'rvo', 'stargardt'],
                                      n_frames=model_config['num_frames'],
                                      model_type=model_config['model_type'],
                                      gray_scale=gray_scale,
                                      logger=logger)
            val_set = Multiclass_ds(loader_type=dataset_info['loader_type'],
                                    annotation_path=where + '/' + dataset_info['annotation_path_val'],
                                    image_size=model_config['image_size'],
                                    categories=['cnv1', 'cnv2', 'cnv3',
                                                'dme', 'ga', 'healthy',
                                                'iamd', 'rvo', 'stargardt'],
                                    model_type=model_config['model_type'], n_frames=model_config['num_frames'],
                                    gray_scale=gray_scale,
                                    logger=logger)
        elif dataset_info['dataset_name'] == 'OLIVES':
            logger.info('Load OLIVES set for training ...')
            train_set = OLIVES_ds(loader_type=dataset_info['loader_type'],
                                  annotation_path=where + '/' + dataset_info['annotation_path_train'],
                                  augment=True,
                                  augmentation_list=available_augmentations,
                                  image_size=model_config['image_size'],
                                  categories=['dr', 'dme'],
                                  n_frames=model_config['num_frames'],
                                  model_type=model_config['model_type'],
                                  gray_scale=gray_scale,
                                  logger=logger)
            val_set = OLIVES_ds(loader_type=dataset_info['loader_type'],
                                annotation_path=where + '/' + dataset_info['annotation_path_val'],
                                image_size=model_config['image_size'],
                                categories=['dr', 'dme'],
                                model_type=model_config['model_type'],
                                n_frames=model_config['num_frames'],
                                gray_scale=gray_scale,
                                logger=logger)
        elif dataset_info['dataset_name'] == 'Duke':
            logger.info('Load Duke set for training ...')
            train_set = Duke_ds(loader_type=dataset_info['loader_type'],
                                annotation_path=where + '/' + dataset_info['annotation_path_train'],
                                augment=True,
                                augmentation_list=available_augmentations,
                                image_size=model_config['image_size'],
                                categories=["AMD", "Normal"],
                                model_type=model_config['model_type'],
                                gray_scale=gray_scale, n_frames=model_config['num_frames'],
                                logger=logger, where=where)
            val_set = Duke_ds(loader_type=dataset_info['loader_type'],
                              annotation_path=where + '/' + dataset_info['annotation_path_val'],
                              image_size=model_config['image_size'],
                              categories=["AMD", "Normal"],
                              model_type=model_config['model_type'],
                              gray_scale=gray_scale, n_frames=model_config['num_frames'],
                              logger=logger, where=where)

        logger.info('[INFO] number of samples in train set: {}'.format(train_set.__len__()))
        logger.info('[INFO] number of samples in val set: {}'.format(val_set.__len__()))

        # initialize the dataloader
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=train_config['batch_size'],
            shuffle=dataset_info['shuffle'],
            num_workers=dataset_info['num_workers'],
            pin_memory=False,
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=train_config['batch_size'],
            shuffle=dataset_info['shuffle'],
            num_workers=dataset_info['num_workers'],
            pin_memory=False,
        )
        # write dataset info in the log file
        logger.info('[INFO] sample distribution in dataset {}:\n # train samples:{}, '
                    ' # validation samples {}'.format(dataset_info['dataset_name'],
                                                      train_set.__len__(),
                                                      val_set.__len__()))
        return train_loader, train_set.__len__(), val_loader, val_set.__len__()
