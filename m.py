import os
import torch


global_config = dict(
    ver="rldm init",
    description="init",
    outcome_root="/kaggle/working",
)
global_config["outcome_dir_root"] = os.path.join(global_config["outcome_root"],
                                                 global_config["ver"])

transformer_config=dict(
    inp_dim = 16,
    dim = 32,
    out_dim = 16,
    num_layers = 4,
    num_heads = 8,
    ff_hidden_dim = 64,
    max_seq_len = 64,
    dropout = 0.0
)


mlp_config=dict(
    in_channels=transformer_config['inp_dim'],
    model_channels=32,
    out_channels=transformer_config['inp_dim'],
    z_channels=transformer_config['out_dim'],
    num_res_blocks=2,
    grad_checkpointing=False
)


diffusion_config=dict(
    num_steps = 64
)

train_config=dict(
    num_fm_per_gd=4,
    max_seq_len=transformer_config['max_seq_len'],
    train_steps=12,
    log_every_n_steps=3,
    eval_every_n_steps=3,
    pretrained=None,
    batch_size=32,
    base_learning_rate=5.0e-5,
    min_learning_rate=4.0e-5,
    use_lr_scheduler=True,
    betas=[0.98, 0.999],
    need_check=False,
    use_ema=False,
    ema_decay=0.9999,
    ema_steps=20000
)
train_config['save']=train_config['train_steps']>0

dataset_paths={'afhq':'/kaggle/input/afhq-512',
               'ffhq':'/kaggle/input/flickrfaceshq-dataset-nvidia-resized-256px',
               'celebahq':'/kaggle/input/celebahq256-images-only',
               'fa':'/kaggle/input/face-attributes-grouped',
               'animestyle':'/kaggle/input/gananime-lite',
               'animefaces':'/kaggle/input/another-anime-face-dataset',
              }

data_config = dict(
    shape=(train_config['batch_size'],
           train_config['max_seq_len'],
           transformer_config['inp_dim']),
    image_size=256,
    batch_size=train_config['batch_size'],
    ae_batch_size=48,
    split=[0.5,0.25,0.25],
    data_paths=dataset_paths,
    enc_path=os.path.join(global_config["outcome_dir_root"], "enc"),
    enc_inp_path='/kaggle/input/sd-vae-ft-ema-f8-256-faces6-enc',
    dataset_names=['afhq', 'ffhq', 'celebahq', 'fa', 'animestyle', 'animefaces'],
    ignored_dataset=['fa'],
    ignored_dataset_ft=['ffhq', 'celebahq', 'animestyle', 'animefaces'],
    valid_dataset_idx=[]
)


from train import train
from build_model import build_model
from data import build_dataset
from model import ARModel
from utils import Logger

import traceback
import shutil

# model = ARModel(transformer_config=transformer_config,
#                 mlp_config=mlp_config,
#                 diffusion_config=diffusion_config,
#                 device=torch.device('cpu')
#                 )

logger = Logger(log_every_n_steps=train_config['log_every_n_steps'],
                log_root=global_config["outcome_dir_root"],
                model_name=global_config['ver']
               )

logger.log_text(str(global_config), "config")
logger.log_text(str(mlp_config), "config", newline=True)
logger.log_text(str(transformer_config), "config", newline=True)
logger.log_text(str(diffusion_config), "config", newline=True)
logger.log_text(str(train_config), "config", newline=True)

torch.manual_seed(42+hash(global_config['ver'])%10000)

train_dataset, val_dataset, test_dataset = build_dataset(data_config)

logger.log_text(str(data_config), "config", newline=True)

model, optim, lr_scheduler = build_model(logger,
                                         transformer_config,
                                         mlp_config,
                                         diffusion_config,
                                         train_config)


try:
    train(model, optim, lr_scheduler, train_config,
          train_dataset, val_dataset, test_dataset, logger)
except Exception as e:
    traceback.print_exc()
    info = traceback.format_exc()
    info = f"Exception: {str(info)} \n"+\
            f"Step: {logger.step}"
    print(info)
    logger.log_text(info, "error")
finally:
    if not any([fn.endswith('.pth') for fn in os.listdir(logger.log_root)]):
        if train_config['save']:
            logger.log_net(model.net.cpu(),f"mar_{logger.step}")
    shutil.make_archive(global_config["outcome_dir_root"],
                        'zip',
                        global_config["outcome_dir_root"])