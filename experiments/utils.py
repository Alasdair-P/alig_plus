import os
import sys
import socket
import torch
import mlogger
import random
import numpy as np
import copy
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('tensor board not found')


def regularization(model, l2):
    reg = 0.5 * l2 * sum([p.data.norm() ** 2 for p in model.parameters()]) if l2 else 0
    return reg


def set_seed(args, print_out=True):
    if args.seed is None:
        np.random.seed(None)
        args.seed = np.random.randint(1e5)
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


def save_state(model, optimizer, filename):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)

def setup_xp(args, model, optimizer):

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        plotter = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(plotter=plotter, **vars(args))

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="training")
    xp.train.loss = mlogger.metric.Average(plotter=plotter, plot_title="Objective", plot_legend="loss")
    xp.train.obj = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(plotter=plotter, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(plotter=plotter, plot_title="Weight-Norm")
    xp.train.grad_norm = mlogger.metric.Average(plotter=plotter, plot_title="Grad-Norm")

    xp.train.step_size = mlogger.metric.Average(plotter=plotter, plot_title="Step-Size", plot_legend="clipped")
    xp.train.step_size_u = mlogger.metric.Average(plotter=plotter, plot_title="Step-Size", plot_legend="unclipped")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='validation')

    xp.max_val = mlogger.metric.Maximum(plotter=plotter, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend='test')

    if args.dataset == "imagenet":
        xp.train.acc5 = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy@5", plot_legend="training")
        xp.val.acc5 = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy@5", plot_legend="validation")
        xp.test.acc5 = mlogger.metric.Average(plotter=plotter, plot_title="Accuracy@5", plot_legend="test")

    if args.visdom:
        visdom_plotter.set_win_opts("Step-Size", {'ytype': 'log'})
        visdom_plotter.set_win_opts("Objective", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        if args.loss == 'map' or args.loss == 'mse':
            xp.max_val = mlogger.metric.Minimum(plotter=plotter, plot_title="Accuracy", plot_legend='best-validation')
            xp.max_val.hook_on_new_min(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))
        else:
            xp.max_val = mlogger.metric.Maximum(plotter=plotter, plot_title="Accuracy", plot_legend='best-validation')
            xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

    return xp

def setup_xp_tb(args, model, optimizer): # please install dev branch of mlogger of tensorboard support then rename this function to setup_xp and delete the other

    env_name = args.xp_name.split('/')[-1]
    if args.visdom:
        visdom_plotter = mlogger.VisdomPlotter({'env': env_name, 'server': args.server, 'port': args.port})
    else:
        visdom_plotter = None

    if args.tensorboard:
        print('args.tensorboard:', args.tensorboard)
        summary_writer = SummaryWriter(log_dir=args.tb_dir)
    else:
        summary_writer = None

    xp = mlogger.Container()

    xp.config = mlogger.Config(visdom_plotter=visdom_plotter, summary_writer=summary_writer)

    vars_ = copy.deepcopy(vars(args))
    # print(vars_)
    for key, value in vars_.items():
        if value is None or type(value) is list:
            vars_[key] = str(value)
    print(vars_)
    # input('press any key')

    xp.config.update(**vars_)

    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="training")
    xp.train.loss = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="loss")
    xp.train.obj = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="objective")
    xp.train.reg = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Objective", plot_legend="regularization")
    xp.train.weight_norm = mlogger.metric.Simple(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Weight-Norm")
    xp.train.grad_norm = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Grad-Norm")

    xp.train.step_size = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Size", plot_legend="clipped")
    xp.train.step_size_u = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Step-Size", plot_legend="unclipped")
    xp.train.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Time", plot_legend='training')

    xp.val = mlogger.Container()
    xp.val.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="validation")
    xp.val.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Time", plot_legend='validation')

    xp.max_val = mlogger.metric.Maximum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')

    xp.test = mlogger.Container()
    xp.test.acc = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(visdom_plotter=visdom_plotter,  summary_writer=summary_writer, plot_title="Time", plot_legend='test')

    if args.dataset == "imagenet":
        xp.train.acc5 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy@5", plot_legend="training")
        xp.val.acc5 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy@5", plot_legend="validation")
        xp.test.acc5 = mlogger.metric.Average(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy@5", plot_legend="test")

    if args.visdom:
        visdom_plotter.set_win_opts("Step-Size", {'ytype': 'log'})
        visdom_plotter.set_win_opts("Objective", {'ytype': 'log'})

    if args.log:
        # log at each epoch
        xp.epoch.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.epoch.hook_on_update(lambda: save_state(model, optimizer, '{}/model.pkl'.format(args.xp_name)))

        # log after final evaluation on test set
        xp.test.acc.hook_on_update(lambda: xp.save_to('{}/results.json'.format(args.xp_name)))
        xp.test.acc.hook_on_update(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

        # save results and model for best validation performance
        if args.loss == 'map' or args.loss == 'mse':
            xp.max_val = mlogger.metric.Minimum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')
            xp.max_val.hook_on_new_min(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))
        else:
            xp.max_val = mlogger.metric.Maximum(visdom_plotter=visdom_plotter, summary_writer=summary_writer, plot_title="Accuracy", plot_legend='best-validation')
            xp.max_val.hook_on_new_max(lambda: save_state(model, optimizer, '{}/best_model.pkl'.format(args.xp_name)))

    return xp

def write_results(args, xp, path):
    file_name = 'results.txt'
    save_path = os.path.join(path, file_name)
    with open(save_path, 'a') as results:
        results.write('dataset,{ds},model,{model},opt,{opt},bs,{bs},eta,{eta},wd,{wd},max_norm,{mn},tr_acc,{tracc:.4f},val_acc,{vacc:.4f},te_acc,{teacc:.4f},name,{xp_name}\n'
                .format(ds=args.dataset,
                        model=args.model,
                        opt=args.opt,
                        bs=args.batch_size,
                        eta=args.eta,
                        wd=args.weight_decay,
                        mn=args.max_norm,
                        tracc=xp.train.acc.value,
                        vacc=xp.max_val.value,
                        teacc=xp.test.acc.value,
                        xp_name=args.xp_name))

def get_acc(args):
    if args.loss == 'mse':
        acc_fuc = squared_loss
    elif args.loss == 'log':
        acc_fuc = logistic_accuracy
    else:
        acc_fuc = accuracy
    return acc_fuc

@torch.autograd.no_grad()
def accuracy(out, targets, topk=1):
    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].reshape(-1).float().sum(0) / out.size(0)

    return 100. * acc

@torch.autograd.no_grad()
def logistic_accuracy(scores, labels):
    labels = labels.view(-1)
    logits = torch.sigmoid(scores).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
    print(pred_labels)
    print(labels)
    input('')
    acc = (pred_labels == labels).float().mean()
    return 100. * acc

@torch.autograd.no_grad()
def squared_loss(logits, labels):
    return torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1), reduction="mean")

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])

            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

class Timer:
    def __init__(self,name):
        self.name = name

    def __enter__(self):
        self.start = time.clock()
        return  self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print('{name} took {int:.4f} seconds'.format(name=self.name, int=self.interval))
        print('------------------------------------------------------------------------')

def smooth_lower_bound(losses, lb, temp=0.1):
    maxl = torch.max(losses, lb)
    minl = torch.min(losses, lb)
    return temp * (torch.log(torch.exp(minl-maxl/temp) + 1) + maxl/temp)

def loss_function(model, images, labels, loss, l2, backwards=False):
    logits = model(images)
    criterion = loss
    loss_value = criterion(logits, labels.view(-1)).mean()
    if l2:
        loss_value += 0.5 * l2 * torch.sqrt(sum(p.data.norm() ** 2 for p in model.parameters()))

    if backwards and loss.requires_grad:
        loss_value.backward()

    return loss_value

def pal_loss_fn(backward=True):
    out_ = net(inputs)
    loss_ = criterion(out_, targets)
    if backward:
        loss_.backward()
    return loss_, out_

