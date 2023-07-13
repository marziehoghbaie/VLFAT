def ViT_create_model(model_config, model_layout, logger):

    if model_config['model_type'] == 'FEViViT':  # first spatial within each frame and
        # then temporal attention among all the video frames
        from model_zoo.ViViTs.models import FEViViT
        model = FEViViT(image_size=model_config['image_size'],
                        patch_size=model_layout['patch_size'],
                        num_classes=model_config['num_classes'],
                        num_frames=model_config['num_frames'],
                        dim=model_layout['dim'],
                        depth=model_layout['depth'],
                        heads=model_layout['heads'],
                        pool=model_layout['pool'],
                        in_channels=model_config['channels'],
                        dim_head=model_layout['dim_head'],
                        dropout=model_layout['dropout'],
                        emb_dropout=model_layout['emb_dropout'],
                        scale_dim=model_layout['scale_dim'],
                        with_pose=model_layout['with_pose'])
    elif model_config['model_type'] == 'FSAViViT':  # first spatial and then temporal attention
        # but it is repeated in each block
        from model_zoo.ViViTs.models import FSAViViT
        model = FSAViViT(t=model_config['num_frames'],
                         h=model_config['image_size'],
                         w=model_config['image_size'],
                         patch_t=model_layout['patch_t'],
                         patch_h=model_layout['patch_h'],
                         patch_w=model_layout['patch_w'],
                         num_classes=model_config['num_classes'],
                         dim=model_layout['dim'],
                         depth=model_layout['depth'],
                         heads=model_layout['heads'],
                         mlp_dim=model_layout['mlp_dim'],
                         dim_head=model_layout['dim_head'],
                         channels=model_config['channels'],
                         mode=model_layout['mode'],
                         emb_dropout=model_layout['emb_dropout'],
                         dropout=model_layout['dropout'],
                         with_pose=model_layout['with_pose'])
    elif model_config['model_type'] == 'ResNet':
        from model_zoo.feature_extrc.models import CNN
        model = CNN(pretrained=model_layout['pretrained'],
                    model_type=model_layout['model_type'],
                    dim=model_layout['embed_dim'],
                    num_classes=model_config['num_classes'],
                    aggregation=model_config['aggregation_type'])

    elif model_config['model_type'] == 'ViTTD_CNV1D':
        from model_zoo.feature_extrc.models import ViTTD_CNV1D
        model = ViTTD_CNV1D(model_type=model_layout['model_type'],
                            pretrained=model_layout['pretrained'],
                            num_classes=model_config['num_classes'],
                            dim=model_layout['embedd_dim'],
                            n_frames=model_config['num_frames'],
                            logger=logger)
    elif model_config['model_type'] == 'ViT_VaR':
        from model_zoo.feature_extrc.models import ViT_VaR
        model = ViT_VaR(model_type=model_layout['model_type'],
                        pretrained=model_layout['pretrained'],
                        interpolation_type=model_layout['interpolation_type'],
                        num_classes=model_config['num_classes'],
                        n_frames=model_config['num_frames'],
                        dim=model_layout['embedd_dim'],
                        logger=logger)
    elif model_config['model_type'] == 'ViT_SinCos':
        from model_zoo.feature_extrc.models import ViT_SinCos
        model = ViT_SinCos(model_type=model_layout['model_type'],
                           pretrained=model_layout['pretrained'],
                           num_classes=model_config['num_classes'],
                           n_frames=model_config['num_frames'],
                           dim=model_layout['embedd_dim'],
                           logger=logger)
    elif model_config['model_type'] == 'ViT_baseline':
        from model_zoo.feature_extrc.models import ViT_baseline
        model = ViT_baseline(model_type=model_layout['model_type'],
                             pretrained=model_layout['pretrained'],
                             num_classes=model_config['num_classes'],
                             n_frames=model_config['num_frames'],
                             dim=model_layout['embedd_dim'],
                             noPE=model_layout['noPE'],
                             logger=logger)

    return model
