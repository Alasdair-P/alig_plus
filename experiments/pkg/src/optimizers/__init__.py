import numpy as np
# import sys
# sys.path.append("..")
from ... import sls
from . import others
# from aalig import AALIG

import torch


class AALIG(torch.optim.Optimizer):

    def __init__(self, params, max_lr=None, data_size=None, epochs=None, momentum=0.9, wd=0.0, mul=True, alig=False, fact2=False, eps=1e-9):
        # if max_lr is not None and max_lr <= 0.0:
            # raise ValueError("Invalid max_lr: {}".format(max_lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(max_lr=max_lr, momentum=momentum, step_size=None)
        super(AALIG, self).__init__(params_list, defaults)

        self.wd = wd
        self.first_update = True
        self.eta_is_zero = (max_lr == 0.0)
        self.max_epochs = epochs
        self.epoch = 0

        self.print = True
        self.print = False
        self.mul = mul
        self.alig = alig
        self.fact2 = fact2
        self.eps = eps

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.device = p.device

        if data_size:
            self.fhat = torch.zeros(data_size, device=self.device) # Estimated optimal value
            print('mean EOV: {m_loss}'.format(m_loss=self.fhat.mean()))
            self.delta = torch.zeros(data_size, device=self.device)
            self.fxbar = torch.ones(data_size, device=self.device) * 1e6 # Loss of sample from best epoch
            self.fx = torch.ones(data_size, device=self.device) * 1e6 # Current loss of samble this epoch
            self.state['lb'] = float(self.fhat.mean())

        self.apply_momentum = self.apply_momentum_standard

    @torch.autograd.no_grad()
    def update_lb(self):
        if self.print:
            print('average fhat', float(self.fhat.mean()), 'fxbar', float(self.fxbar.mean()))


        if self.first_update:
            reached_lb = 0.0
            self.first_update = False
        else:
            reached_lb = (self.fxbar.le(self.fhat)).float()
        if not self.alig:
            if self.mul:
                self.delta = (0.5*self.fxbar - 0.5*self.fhat).abs().mul(1-reached_lb) + self.delta.mul(reached_lb).mul(2)
                self.fhat = (0.5*self.fhat + 0.5*self.fxbar).mul(1-reached_lb) + (self.fhat - 0.25*self.delta).clamp(min=0).mul(reached_lb)
            else:
                self.delta = (0.5*self.fxbar - 0.5*self.fhat).abs().mul(1-reached_lb) + self.delta.mul(reached_lb)
                self.fhat = (0.5*self.fhat + 0.5*self.fxbar).mul(1-reached_lb) + (self.fhat - 0.5*self.delta).clamp(min=0).mul(reached_lb)
            self.state['lb'] = float(self.fhat.mean())

        if self.print:
            print('average fhat', float(self.fhat.mean()), 'fxbar', float(self.fxbar.mean()))
            print('delta', float(self.delta.mean()), self.delta)
            print('average fhat', float(self.fhat.mean()))
            input('press any key')

    @torch.autograd.no_grad()
    def epoch_(self):
        if self.fx.mean() < self.fxbar.mean():
            # if self.global_lb:
                # self.fxbar.fill_(self.fx.mean())
            # else:
            self.fxbar = self.fx
        if self.epoch % (self.max_epochs // 10) == ((self.max_epochs // 10)-1):
            print('updating lower bound')
            self.update_lb()
        print(self.epoch % (self.max_epochs // 10))
        self.epoch += 1

    @torch.autograd.no_grad()
    def compute_step_size(self, losses, lbs):

        # compute squared norm of gradient
        grad_sqrd_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                    continue
                grad_sqrd_norm += p.grad.data.norm() ** 2

        # compute unclipped step-size
        if self.fact2:
            self.step_size_unclipped = float((losses - lbs).mean() / (2 * grad_sqrd_norm + self.eps))
        else:
            self.step_size_unclipped = float((losses - lbs).mean() / (grad_sqrd_norm + self.eps))

        if self.print:
            print('losses', losses, 'lbs', lbs, 'numerator', float((losses - lbs).mean()), '|g|^2', float(grad_sqrd_norm), 'step_size: ', self.step_size_unclipped)
            input('press any key')

        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = max(min(self.step_size_unclipped, group["max_lr"]),0.0)
            else:
                group["step_size"] = max(self.step_size_unclipped,0.0)

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))
        self.state["step_size"] = self.step_size

    @torch.autograd.no_grad()
    def step(self, closure):
        idx, losses = closure()
        self.fx[idx] = losses
        fhat = self.fhat[idx]
        if self.print:
            print('idx', idx, 'losses', losses, 'fbar', fhat)
        self.compute_step_size(losses, fhat)

        for group in self.param_groups:
            step_size = group["step_size"]
            momentum = group["momentum"]
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad+p.mul(self.wd)
                p.add_(-step_size, grad)
                # Nesterov momentum
                if momentum:
                    self.apply_momentum(p, step_size, momentum)

    @torch.autograd.no_grad()
    def apply_momentum_standard(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(-step_size, p.grad)
        p.add_(momentum, buffer)



def get_optimizer(opt, params, max_epochs, train_len, n_batches_per_epoch=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    # ===============================================
    # our optimizers   
    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch    
    if opt_name == "sgd_armijo":
        if opt_dict.get("infer_c"):
            c = (1e-3) * np.sqrt(n_batches_per_epoch)
        
        opt = sls.Sls(params,
                    c = opt_dict.get("c") or 0.1,
                    n_batches_per_epoch=n_batches_per_epoch,
                    line_search_fn="armijo")

    elif opt_name == "sgd_goldstein":
        opt = sls.Sls(params, 
                      c=opt_dict.get("c") or 0.1,
                      reset_option=opt_dict.get("reset_option") or 0,
                      n_batches_per_epoch=n_batches_per_epoch,
                      line_search_fn="goldstein")

    elif opt_name == "sgd_nesterov":
        opt = sls.SlsAcc(params, 
                            acceleration_method="nesterov")

    elif opt_name == "sgd_polyak":
        opt = sls.SlsAcc(params, 
                         c=opt_dict.get("c") or 0.1,
                         acceleration_method="polyak")
    
    elif opt_name == "seg":
        opt = sls.SlsEg(params, n_batches_per_epoch=n_batches_per_epoch)


    # ===============================================
    # others
    elif opt_name == "adam":
        opt = torch.optim.Adam(params)

    elif opt_name == "adagrad":
        opt = torch.optim.Adagrad(params)

    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=1e-3)

    elif opt_name == 'alimn':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs)
    elif opt_name == 'alian':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, mul=False)
    elif opt_name == 'alimo':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, momentum=0.0)
    elif opt_name == 'alimo2':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, momentum=0.0, fact2=True)
    elif opt_name == 'aliao':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, momentum=0.0, mul=False)

    elif opt_name == 'align':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, momentum=0.9, alig=True)
    elif opt_name == 'aligo':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, momentum=0.0, alig=True)
    elif opt_name == 'aligo2':
        opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=train_len, epochs=max_epochs, momentum=0.0, alig=True, fact2=True)
    # elif opt_name == 'ali0n':
        # opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=800, epochs=max_epochs, momentum=0.9, alig=True, eps=0.0)
    # elif opt_name == 'ali0o':
        # opt = AALIG(params, max_lr=opt_dict.get("max_lr"), data_size=800, epochs=max_epochs, momentum=0.0, alig=True, eps=0.0)

    elif opt_name == 'adagrad':
        opt = torch.optim.Adagrad(params)

    elif opt_name == 'rms':
        opt = torch.optim.RMSprop(params)

    elif opt_name == 'svrg':
        opt = SVRG(params)

    elif opt_name == 'adabound':
        opt = others.AdaBound(params)
        print('Running AdaBound..')

    elif opt_name == 'amsbound':
        opt = others.AdaBound(params, amsbound=True)

    elif opt_name == 'coin':
        opt = others.CocobBackprop(params)

    elif opt_name == 'l4':
        params = list(params)
        # base_opt = torch.optim.Adam(params)
        base_opt = torch.optim.SGD(params, lr=0.01, momentum=0.5)
        opt = others.L4(params, base_opt)

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt
