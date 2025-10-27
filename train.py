import torch
import numpy as np
from utils import Logger, check_ae


def train(model,
          optim,
          lr_scheduler,
          train_config,
          train_dataset,
          val_dataset,
          test_dataset,
          logger: Logger):
    if train_config['train_steps']<=0:
        model.eval()
        final_eval_generation(model, train_config, logger, verbose=train_config['train_steps']==0)
        return
    
    logger.train_start()
    for x0 in train_dataset:
        model.train()
        x0 = x0.to(model.device)
        x0 = model.preprocess(x0)
        optim.zero_grad()
        loss = model.train_step(x0)
        loss.backward()
        optim.step()
        
        logger.train_step(loss.detach().cpu().item())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps'] == 0:
            model.eval()
            test(model, logger, val_dataset, num_test_steps=1000, is_eval=True)
        if logger.step == train_config['train_steps']:
            model.eval()
            test_gen(model,test_dataset,logger,num=10)
            logger.train_end()
            break
    
    loss1=test(model, logger, test_dataset, num_test_steps=1000)
    
    if train_config['save']:
        logger.log_net(model.cpu(),f"mar_{logger.step}_{logger.model_name}")
    return

@torch.no_grad()
def test_gen(model, test_dataset, logger, num=10):
    x0s=next(iter(test_dataset))[:4]
    x0s= x0s.to(model.device)
    x0s = model.preprocess(x0s)
    out = []
    print("generating")
    for _ in range(num):
        res = model.gen(x0s[:, :-33],scope=32)
        out.append(res.cpu().numpy())
    logger.test_gen(x0s.cpu().numpy(), out=out, idx=32)
    

@torch.no_grad()
def eval_generation(model, train_config, logger):
    logger.generation_start()
    for cls in range(5):
        if not cls in train_config['valid_dataset_idx']:
            continue
        imgs = model.conditional_generation(cls,
                                            guidance_scale=1,
                                            batch_size=9,
                                            use_2nd_order=False,
                                            n_steps=512,
                                            )
        logger.log_images(imgs, 3, 3, f"step_{logger.step}_cls_{cls}_cfg_1_step_512")
    logger.generation_end()
    logger.train_resume()

@torch.no_grad()
def final_eval_generation(model, train_config, logger, verbose=False):
    logger.generation_start()
    if verbose:
        cfg_=[1.5, 2, 3]
        cls__=[0,2,3]
        cls_=[]
        for cls in cls__:
            if cls in train_config['valid_dataset_idx']:
                cls_.append(cls)
        for cfg in cfg_:
            for cls in cls_:
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    cfg_zero_star=(True, False),
                                                    )
                logger.log_images(imgs, 4, 4, f"cls_{cls}_cfg_{cfg}_step_512_czs1")
        for cfg in cfg_:
            for cls in cls_:
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=1024,
                                                    cfg_zero_star=(True, False),
                                                    )
                logger.log_images(imgs, 4, 4, f"cls_{cls}_cfg_{cfg}_step_1024_czs1")
        torch.cuda.empty_cache()
        for cfg in cfg_*2:
            for cls in cls_:
                imgs = model.conditional_generation_with_middle_steps(cls,
                                                                    cfg,
                                                                    use_2nd_order=False,
                                                                    batch_size=4,
                                                                    n_steps=512,
                                                                    n_middle_steps=7,
                                                                    cfg_zero_star=(True,False)
                                                                    )
                logger.log_images(imgs, 4, 8, f"cls_{cls}_cfg_{cfg}_step_512_mid_czs1_pred")
    else:
        for cfg in [1,1,1]:
            for cls in range(5):
                if not cls in train_config['valid_dataset_idx']:
                    continue
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    12,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    cfg_zero_star=True
                                                    )
                logger.log_images(imgs, 4, 3, f"czs_cls_{cls}_cfg_{cfg}_step_512")
    logger.generation_end()

@torch.no_grad()
def test(model,
         logger,
         dataset,
         num_test_steps=1000,
         is_eval=False
         ):
    model.eval()
    acc_loss = []
    step = 0
    for x0 in dataset:
        step += 1
        x0 = x0.to(model.device)
        x0 = model.preprocess(x0)
        loss = model.train_step(x0)
        acc_loss.append(loss.cpu().item())
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss)
    mode="Eval" if is_eval else "Test"
    info = f"{mode}\n" \
           + f"loss:{acc_loss.mean():.4f}+-{acc_loss.std():.4f}\n" 
    print(info)
    logger.log_text(info, "train_log", newline=True)
    return acc_loss.mean()
