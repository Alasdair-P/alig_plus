import torch
import torch.utils.data as data

class IndexedDataset(data.Dataset):
    def __init__(self, dataset):
        self._dataset = dataset
    def __getitem__(self, idx):
        return idx, self._dataset[idx]
    def __len__(self):
        return self._dataset.__len__()


class AligPlus(torch.optim.Optimizer):
    r"""
    Implements the Adaptive ALI-G (ALIG+) algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): maximal learning rate 0.1 works well for more porblems
        data_size (int): number of elements in training set
	epochs (int): total number of epochs of training
        weight_dcay (float, optional): weight decay amount
        momentum (float, optional): momentum factor (default: 0.9)
    Example:
        >>> optimizer = AligPlus(model.parameters(), args.lr, args.train_size, args.epochs momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_value = loss_fni(reduction='none')(model(input), target)
        >>> loss_value.backward()
        >>> optimizer.step(lambda: (idx,losses))
	>>>
        >>> after each epoch call optimizer.epoch_()
	.. note::
        In order to compute the step-size, this optimizer requires a closure containing the index and loss i
        of each sample in the at every step. The IndexedDataset above can be used to wrap non-indexed training sets.
    """
    def __init__(self, params, lr, data_size, epochs, weight_decay=0.0, momentum=0.9, K=5):
        if lr < 0.0:
            raise ValueError("Invalid max_lr: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(max_lr=lr, momentum=momentum, step_size=None, wd=weight_decay)
        super(AligPlus, self).__init__(params_list, defaults)
        print('creating aalig optimiser')

        self.k = K

        self.max_epochs = epochs
        self.epoch = 0

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.aovs = torch.zeros(data_size, device=p.device) # AOVs
        self.aov_minus_last = torch.zeros(data_size, device=p.device) #  = A0V_{k} - AOV_{k-1}
        self.lbars = torch.ones(data_size, device=p.device) * 1e6 # Loss of each sample from best epoch
        self.ls = torch.ones(data_size, device=p.device) * 1e6 # Current loss of sample this epoch

    @torch.autograd.no_grad()
    def update_lb(self):
        print('updating aovs')
        reached_aov = (self.lbars.le(self.aovs)).float() # zero one mask for if a give aov has been reached
        self.aov_minus_last = (0.5*self.lbars - 0.5*self.aovs).clamp(min=0).mul(1-reached_aov) + self.aov_minus_last.mul(reached_aov)
        self.aovs = (0.5*self.aovs + 0.5*self.lbars).mul(1-reached_aov) + (self.aovs - 0.5*self.aov_minus_last).clamp(min=0).mul(reached_aov)

    @torch.autograd.no_grad()
    def epoch_(self):
        """
        method to be called after each epoch
        """
        if self.ls.mean() < self.lbars.mean(): # approximate l_z(\bar{\w}) where, \bar{\w} = min_{t\in{1,..,T}}(\w_t)
            self.lbars = self.ls

        if self.epoch % (self.max_epochs // self.k) == ((self.max_epochs // self.k)-1):
            self.update_lb()

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

        self.step_size_unclipped = float((losses - lbs).mean() / (grad_sqrd_norm + 1e-9))

        # compute effective step-size (clipped)
        for group in self.param_groups:
            if group["max_lr"] is not None:
                group["step_size"] = max(min(self.step_size_unclipped, group["max_lr"]),0.0)
            else:
                group["step_size"] = max(self.step_size_unclipped,0.0)

        # average step size for monitoring
        self.step_size = sum([g["step_size"] for g in self.param_groups]) / float(len(self.param_groups))

    @torch.autograd.no_grad()
    def step(self, closure):
        idx, losses = closure()

        self.ls[idx] = losses
        aovs = self.aovs[idx]

        self.compute_step_size(losses, aovs)

        for group in self.param_groups:
            step_size = group["step_size"]
            momentum = group["momentum"]
            wd = group['wd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad + p.mul(wd)
                with torch.no_grad():
                    p.add_(grad,alpha=-step_size)
                    if momentum:
                        self.apply_momentum(p, step_size, momentum)

    @torch.autograd.no_grad()
    def apply_momentum(self, p, step_size, momentum):
        buffer = self.state[p]['momentum_buffer']
        buffer.mul_(momentum).add_(p.grad,alpha=-step_size)
        p.add_(buffer, alpha=momentum)
