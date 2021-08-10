import os
import argparse
import logging

def parser():

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--data_path', default='~/data/cifar', type=str, help='path for input data')
    parser.add_argument('--num_epoches', default=200, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of workers in dataloader')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--temp', default=0.1, type=float, help='temperature')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--adversarial_test', '-t', action='store_true', help='adversarial test')
    parser.add_argument('--gpu', '-g', default='0,1,2,3,4,5,6,7', help='which gpu to use')
    parser.add_argument('--mart', default=True, action='store_true', help='use mart loss')
    parser.add_argument('--log_root', default='./log', help='the directory to save the logs')
    parser.add_argument('--ckpt_root', default='./checkpoint', help='the directory to save the ckeckpoints')
    parser.add_argument('--nat_init', default=True, action='store_true', help='initialize with pretrained model')
    parser.add_argument('--nat_root', default='../../natural-training-lbs05-100150/cifar10', help='the directory for the natural ckeckpoints')
    parser.add_argument('--nat_file', default='ckpt_latest.t7', help='the name the natural ckeckpoints')

    args = parser.parse_args()

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

