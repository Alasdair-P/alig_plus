# top-import for cuda device initialization
from cuda import set_cuda

import mlogger
import torch

from cli import parse_command
from loss import get_loss
from utils import setup_xp, set_seed, save_state, write_results
from data import get_data_loaders
from models import get_model, load_best_model
from optim import get_optimizer, decay_optimizer
from epoch import train, test


def main(args):
    set_cuda(args)
    set_seed(args)

    loader_train, loader_val, loader_test = get_data_loaders(args)
    loss = get_loss(args)
    model = get_model(args)

    print([module for module in model.modules() if type(module) != torch.nn.Sequential][0])
    optimizer = get_optimizer(args, model, loss, parameters=model.parameters())
    xp = setup_xp(args, model, optimizer)
    for i in range(args.epochs):
        xp.epoch.update(i)
        train(model, loss, optimizer, loader_train, args, xp)
        test(model, optimizer, loader_val, args, xp)
        if 'alig_plus' in args.opt:
            optimizer.epoch_()

        if (i + 1) in args.T:
            decay_optimizer(args, optimizer, args.decay_factor)
            if 'alig_plus' in args.opt:
                optimizer.update_lb()

    load_best_model(model, '{}/best_model.pkl'.format(args.xp_name))
    test(model, optimizer, loader_val, args, xp)
    test(model, optimizer, loader_test, args, xp)
    write_results(args, xp, '.')


if __name__ == '__main__':
    args = parse_command()
    with mlogger.stdout_to("{}/log.txt".format(args.xp_name), enabled=args.log):
        main(args)
