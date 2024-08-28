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
        steps_per_epoch=4324,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_target_lr = warmup_target_lr
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.steps_per_epoch = steps_per_epoch
        super(ExponentialDecayWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        is_first_epoch = self._step_count - 1 < self.steps_per_epoch * self.warmup_epochs
        if is_first_epoch:
            # Warmup phase
            lrs = [
                self.warmup_target_lr * (self._step_count) / self.steps_per_epoch
                for _ in self.base_lrs
            ]
        else:
            # Exponential decay phase
            decay_epochs = self.total_epochs - self.warmup_epochs
            decay_ratio = (self.final_lr / self.initial_lr) ** (1 / decay_epochs)
            exponential_decay_epochs = ((self._step_count - self.steps_per_epoch * self.warmup_epochs) / self.steps_per_epoch) # epochs - warmup epochs
            lrs = [
                self.initial_lr
                * (decay_ratio ** (exponential_decay_epochs))
                for _ in self.base_lrs
            ]
        return lrs
