import torch.optim
from alig.th import AliG, Yogi, AdamW
from alig_plus import AligPlus
from global_alig_plus import GlobalAligPlus
from pal import PalOptimizer
#from alig2 import AliG2
from alig.th.projection import l2_projection
from pal import PalOptimizer
from adamp import AdamP
from pkg import sls
from pkg.src.optimizers import others
import sps

def get_optimizer(args, model, loss, parameters):
    print('batches per epoch: ', args.n_batches_per_epoch)
    parameters = list(parameters)
    data_size = (args.train_size,)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay, amsgrad=True)
    elif args.opt == 'sps':
        optimizer = sps.Sps(parameters, n_batches_per_epoch=args.n_batches_per_epoch, eta_max=args.eta)
    elif args.opt == "yogi":
        optimizer = Yogi(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = AdamW(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == 'bpgrad':
        optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'alig_plus':
        optimizer = AligPlus(parameters, lr=args.eta, data_size=data_size, weight_decay=args.weight_decay,
                          epochs=args.epochs, momentum=args.momentum, K=args.K)
    elif args.opt == 'global_alig_plus':
        optimizer = GlobalAligPlus(parameters, lr=args.eta, data_size=data_size, weight_decay=args.weight_decay,
                          epochs=args.epochs, momentum=args.momentum, K=args.K)
    elif args.opt == 'galig_plus':
        optimizer = GlobalAligPlus(parameters, lr=args.eta, data_size=data_size, weight_decay=args.weight_decay,
                          epochs=args.epochs, momentum=args.momentum, K=args.K)
    elif args.opt == 'alig':
        optimizer = AliG(parameters, max_lr=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm))
    elif args.opt == 'sps':
        optimizer = Sps(parameters, n_batches_per_epoch=args.n_batches_per_epoch, eta_max=args.eta)
    elif args.opt == 'bpgrad':
        optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'l4adam':
        optimizer = L4Adam(parameters, weight_decay=args.weight_decay)
    elif args.opt == 'l4mom':
        optimizer = L4Mom(parameters, weight_decay=args.weight_decay)
    elif args.opt == 'alig':
        optimizer = AliG(parameters, max_lr=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm))
    elif args.opt == "sgd_armijo":
        optimizer = sls.Sls(parameters,
                    c=0.1,
                    n_batches_per_epoch=args.n_batches_per_epoch,
                    line_search_fn="armijo")
    elif args.opt == "sgd_goldstein":
        optimizer = sls.Sls(parameters,
                      c=0.1,
                      reset_option=0,
                      eta_max=args.eta,
                      n_batches_per_epoch=args.n_batches_per_epoch,
                      line_search_fn="goldstein")
    elif args.opt == "sgd_polyak":
        optimizer = sls.SlsAcc(parameters,
                         c=0.1,
                         acceleration_method="polyak")
    elif args.opt == "pal":
        optimizer = PalOptimizer(parameters, None, measuring_step_size=0.1, max_step_size=args.eta,
                         update_step_adaptation=1,
                         direction_adaptation_factor=0.4, is_plot=False,
                         plot_step_interval=100, save_dir="./lines/")
    elif args.opt == 'adabound':
        optimizer = others.AdaBound(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == 'coin':
        optimizer = others.CocobBackprop(parameters)
    elif args.opt == 'adamp':
        optimizer = AdamP(parameters, lr=args.eta, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.step_size = args.eta or 0.0
    optimizer.step_size_unclipped = args.eta or 0.0
    optimizer.momentum = args.momentum

    if args.load_opt:
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))

    return optimizer

def decay_optimizer(args, optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor

            optimizer.step_size = optimizer.param_groups[0]['lr']
            optimizer.step_size_unclipped = optimizer.param_groups[0]['lr']
        else:
            print('decay learning rate only supported for SGD')
            # raise ValueError
