import math

import torch


def make_scheduler(opt, scfg):
    iters = int(scfg["iters"])
    start = int(scfg["anneal_start"])
    floor = float(scfg["anneal_min_factor"])

    def factor(it):
        if it < start or iters <= start:
            return 1.0
        prog = (it - start) / max(iters - start, 1)          # 0 -> 1
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)
