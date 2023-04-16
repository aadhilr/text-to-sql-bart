import argparse
import json
import os


def read_arguments_train():
    parser = argparse.ArgumentParser(description="Run training with following arguments")

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--data_set', default='spider', type=str)

    # encoder configuration
    parser.add_argument('--encoder_pretrained_model', default='facebook/bart-base', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int)

    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr_transformer', default=2e-5, type=float)

    args = parser.parse_args()

    args.data_dir = os.path.join('data', args.data_set)
    args.model_output_dir = 'experiments'

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args


def read_arguments_evaluation():
    parser = argparse.ArgumentParser(description="Run evaluation with following arguments")

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--data_set', default='spider', type=str)

    parser.add_argument('--encoder_pretrained_model', default='facebook/bart-base', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int)

    parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')

    args = parser.parse_args()
    args.data_dir = os.path.join('data', args.data_set)
    args.model_output_dir = 'experiments'

    for argument in vars(args):
        print("argument: {}={}".format(argument, getattr(args, argument)))

    return args
