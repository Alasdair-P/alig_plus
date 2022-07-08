try:
    import torch
    import numpy as np
except ImportError:
    raise ImportError("PyTorch is not installed, impossible to import `alig.th.AliG`")


class AALIG(torch.optim.Optimizer):

    def __init__(self, params, max_lr=None, data_size=None, epochs=None, momentum=0.9, wd=None):
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

        self.delta = (0.5*self.fxbar - 0.5*self.fhat).abs().mul(1-reached_lb) + self.delta.mul(reached_lb).mul(2)
        self.fhat = (0.5*self.fhat + 0.5*self.fxbar).mul(1-reached_lb) + (self.fhat - 0.25*self.delta).clamp(min=0).mul(reached_lb)

        if self.print:
            print('average fhat', float(self.fhat.mean()), 'fxbar', float(self.fxbar.mean()))
            print('delta', float(self.delta.mean()), self.delta)
            print('average fhat', float(self.fhat.mean()))
            input('press any key')


    @torch.autograd.no_grad()
    def epoch(self):
        if self.fx.mean() < self.fxbar.mean():
            if self.global_lb:
                self.fxbar.fill_(self.fx.mean())
            else:
                self.fxbar = self.fx
        if self.epoch % self.max_epoch // 10 == ((self.max_epoch // 10)-1):
            print('updating lower bound')
            self.update_lb()
        print(self.epoch % self.max_epoch // 10)
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
        self.step_size_unclipped = float((losses - lbs).mean() / (grad_sqrd_norm + 1e-4))

        if self.sgd_mode:
            for group in self.param_groups:
                self.step_size_unclipped = group["max_lr"]

        if self.print:
            print('losses', losses, 'lbs', lbs, 'numerator', float((losses - lbs).mean()), '|g|^2', float(grad_sqrd_norm), 'step_size: ', self.step_size_unclipped)
            input('press any key')

        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = max(min(self.step_size_unclipped, group["max_lr"]),0.0)
            else:
                # print('max_lr is None')
                group["step_size"] = self.step_size_unclipped

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))

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


if __name__ == "__main__":
    opt = AliG2(torch.Tensor([10]), max_lr=None, data_size= 1 momentum=0, wd=0)

