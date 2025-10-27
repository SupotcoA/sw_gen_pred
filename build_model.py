import torch
from model import ARModel
from utils import calculate_num_params

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineSchedulerWithWarmup(_LRScheduler):
    """
    Cosine learning rate scheduler with warmup.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of warmup epochs.
        max_epochs (int): Maximum number of epochs (including warmup).
        warmup_start_lr (float, optional): Initial learning rate during warmup. Default: 0.
        max_lr (float): Maximum learning rate after warmup.
        min_lr (float, optional): Minimum learning rate after cosine decay. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """
    
    def __init__(self, 
                 optimizer: Optimizer, 
                 warmup_epochs: int, 
                 max_epochs: int, 
                 max_lr: float,
                 warmup_start_lr: float = 0., 
                 min_lr: float = 0.,
                 last_epoch: int = -1):
        
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cosine_epochs = max(max_epochs - warmup_epochs, 1)  # Ensure at least 1
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
            
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.last_epoch / max(self.warmup_epochs, 1)
            return [self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * progress 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / self.cosine_epochs
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_decay 
                    for base_lr in self.base_lrs]


def build_model(logger,
                transformer_config,
                mlp_config,
                diffusion_config,
                train_config):
    model = ARModel(transformer_config,
                    mlp_config,
                    diffusion_config,
                    train_config,
                    )
    t_params = calculate_num_params(model.transformer)
    mlp_params = calculate_num_params(model.mlp)
    t_trainable_params = calculate_num_params(model.transformer, trainable_only=True)
    info = f"T params: {t_params:,}, MLP params: {mlp_params:,}, TTrainable: {t_trainable_params:,}"
    print(info)
    logger.log_text(info, "config", newline=True)

    if train_config['pretrained']:
        sd = torch.load(train_config['pretrained'], map_location=torch.device('cpu'))
        model.net.load_state_dict(sd, strict=True)
    if torch.cuda.is_available():
        model.cuda()
        print("running on cuda")
    else:
        print("running on cpu!")
    # TODO: use AdamW
    optim = torch.optim.Adam(model.parameters(),
                             lr=train_config['base_learning_rate'],
                             betas=train_config['betas'])
    if train_config['use_lr_scheduler']:
        lr_scheduler = CosineSchedulerWithWarmup(optimizer=optim,
                                                 max_epochs=train_config['train_steps'],
                                                 warmup_epochs=train_config['warmup_steps'],
                                                 max_lr=train_config['base_learning_rate'],
                                                 warmup_start_lr=train_config['base_learning_rate']/10,
                                                 min_lr=train_config['min_learning_rate'])
    else:
        lr_scheduler = None
    return model.eval(), optim, lr_scheduler
