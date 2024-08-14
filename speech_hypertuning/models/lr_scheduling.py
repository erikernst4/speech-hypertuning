from torch.optim.lr_scheduler import _LRScheduler


class ExponentialDecayWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr,
        final_lr,
        warmup_epochs,
        warmup_target_lr,
        total_epochs,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_target_lr = warmup_target_lr
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super(ExponentialDecayWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                self.warmup_target_lr * (self.last_epoch + 1) / self.warmup_epochs
                for _ in self.base_lrs
            ]
        else:
            # Exponential decay phase
            decay_epochs = self.total_epochs - self.warmup_epochs
            decay_ratio = (self.final_lr / self.initial_lr) ** (1 / decay_epochs)
            return [
                self.initial_lr
                * (decay_ratio ** (self.last_epoch - self.warmup_epochs))
                for _ in self.base_lrs
            ]
