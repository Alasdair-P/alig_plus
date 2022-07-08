import torch
import os
import numpy as np
import time
from tqdm import tqdm
from utils import get_acc, regularization, smooth_lower_bound, loss_function

def train(model, loss, optimizer, loader, args, xp):
    model.train()

    for metric in xp.train.metrics():
        metric.reset()

    for idx, data in tqdm(loader, disable=not args.tqdm, desc='Train Epoch',
                                               leave=False, total=len(loader)):
        x_, y = data
        if isinstance(x_,dict):
            transforms, x = x_['trans'], x_['image']
        else:
            x = x_
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)

        if (args.opt == "sgd_armijo") or (args.opt == "sgd_goldstein") or (args.opt == "sgd_polyak"):
            optimizer.zero_grad()
            scores = model(x)
            raw_loss = loss(scores, y).mean().clone()
            closure = lambda : loss_function(model, x, y, loss, args.l2_reg, backwards=False)
            optimizer.step(closure)

        elif (args.opt == "pal"):
            optimizer.zero_grad()

            def loss_fn(backward=True):
                scores = model(x)
                raw_loss = loss(scores, y).mean().clone()
                if args.l2_reg:
                    raw_loss += 0.5 * args.l2_reg * torch.sqrt(sum(p.data.norm() ** 2 for p in model.parameters()))
                if backward:
                    raw_loss.backward()
                return raw_loss, scores

            raw_loss, scores, _ = optimizer.step(loss_fn)
        else:

            # forward pass
            scores = model(x)

            # compute the loss function, possibly using smoothing
            # with set_smoothing_enabled(args.smooth_svm):
            losses = loss(scores, y)
            if args.l2_reg:
                losses += 0.5 * args.l2_reg * torch.sqrt(sum(p.data.norm() ** 2 for p in model.parameters()))
            raw_loss = losses.mean().clone()

            if  args.temp:
                clipped_losses = smooth_lower_bound(losses, optimizer.fhat[idx], args.temp)
            else:
                clipped_losses = losses

            loss_value = clipped_losses.mean()

            # backward pass
            optimizer.zero_grad()
            loss_value.backward()
                # optimization step
            if 'alig_plus' in args.opt:
                optimizer.step(lambda: (idx,losses))
            elif args.opt == 'sps':
                optimizer.step(loss = loss_value)
            else:
                optimizer.step(lambda: loss_value)

        if 'sbd' in args.opt and not optimizer.n == 1:
            continue

        # monitoring
        batch_size = x.size(0)
        accuracy_fuc = get_acc(args)
        xp.train.acc.update(accuracy_fuc(scores, y), weighting=batch_size)
        xp.train.loss.update(raw_loss, weighting=batch_size)
        xp.train.step_size.update(optimizer.step_size, weighting=batch_size)
        xp.train.step_size_u.update(optimizer.step_size_unclipped, weighting=batch_size)
        if args.dataset == "imagenet":
            xp.train.acc5.update(accuracy_fuc(scores, y, topk=5), weighting=batch_size)

    xp.train.grad_norm.update(torch.sqrt(sum(p.grad.data.norm() ** 2  for p in model.parameters())))
    xp.train.weight_norm.update(torch.sqrt(sum(p.data.norm() ** 2 for p in model.parameters())))
    xp.train.reg.update(0.5 * (args.weight_decay or args.l2_reg) * xp.train.weight_norm.value ** 2)
    xp.train.obj.update(xp.train.reg.value + xp.train.loss.value)
    xp.train.timer.update()

    print('\nEpoch: [{0}] (Train) \t'
          '({timer:.2f}s) \t'
          'Obj {obj:.3f}\t'
          'Loss {loss:.3f}\t'
          'Acc {acc:.2f}%\t'
          .format(int(xp.epoch.value),
                  timer=xp.train.timer.value,
                  acc=xp.train.acc.value,
                  obj=xp.train.obj.value,
                  loss=xp.train.loss.value))

    for metric in xp.train.metrics():
        metric.log(time=xp.epoch.value)

@torch.autograd.no_grad()
def test(model, optimizer, loader, args, xp):
    model.eval()

    if loader.tag == 'val':
        xp_group = xp.val
    else:
        xp_group = xp.test

    for metric in xp_group.metrics():
        metric.reset()

    for x, y in tqdm(loader, disable=not args.tqdm,
                     desc='{} Epoch'.format(loader.tag.title()),
                     leave=False, total=len(loader)):
        (x, y) = (x.cuda(), y.cuda()) if args.cuda else (x, y)
        scores = model(x)

        accuracy_fuc = get_acc(args)
        xp_group.acc.update(accuracy_fuc(scores, y), weighting=x.size(0))
        if args.dataset == "imagenet":
            xp_group.acc5.update(accuracy_fuc(scores, y, topk=5), weighting=x.size(0))

    xp_group.timer.update()

    print('Epoch: [{0}] ({tag})\t'
          '({timer:.3f}s) \t'
          'Obj ----\t'
          'Loss ----\t'
          'Acc {acc:.3f}% \t'
          .format(int(xp.epoch.value),
                  tag=loader.tag.title(),
                  timer=xp_group.timer.value,
                  acc=xp_group.acc.value))

    if loader.tag == 'val':
        xp.max_val.update(xp.val.acc.value).log(time=xp.epoch.value)

    for metric in xp_group.metrics():
        metric.log(time=xp.epoch.value)
