import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('ViT3D', add_help=False)

    #  Config File Path
    parser.add_argument('--config_path', default='',
                        type=str,
                        help='Please specify path to the confile file.')

    return parser