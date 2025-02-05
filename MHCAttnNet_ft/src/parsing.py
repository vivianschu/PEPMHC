import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='PROT_BERT')

    parser.add_argument('--new-split-flag',    default=False,      action="store_true",    help='train_test_split')
    parser.add_argument('--pep-max-len',       default=48,         type=int,               help='peptide max length')
    parser.add_argument('--scratch',                               action='store_true', help='if pretrain pep transformer from sratch (without using protbert pretrained weights)')
    parser.add_argument('--pretrained',                            action='store_true', help='if only use pretrained protbert weights' )
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')


    # Genreal
    # parser.add_argument('--emotion',        default='valence',  type=str,               help='emotion label')
    # parser.add_argument('--scenario',       default=1,          type=int,               help='scenario number')
    # parser.add_argument('--fold',           default=0,          type=int,               help='fold number')

    # parser.add_argument('--modality',       default='ecg',      type=str,               help='bio-sginals')
    # parser.add_argument('--lr',             default=0.0001,     type=float,             help='initial learning rate')
    # parser.add_argument('--epochs',         default=50,         type=int,               help='epochs')
    # parser.add_argument('--batch-size',     default=8,          type=int,               help='batch size')
    # parser.add_argument('--optimizer',      default='sgd',      type=str,               help='optimizer choice')
    # parser.add_argument('--pretraining',    default=False,      action="store_true",    help='pretraining')
    # parser.add_argument('--use-pretrain',   default=False,      action="store_true",    help='use pretrained model')
    # parser.add_argument('--final-flag',     default=False,      action="store_true",    help='obtain final test results')

    # parser.add_argument('--use-scheduler',  default=False,      action="store_true",    help='learning rate scheduler')

    return parser




