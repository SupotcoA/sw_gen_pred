import os
import torch
import numpy as np
from utils import Logger, check_ae
from data_seq import postprocess_data
from probe import pipeline

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
        test(model, logger, train_dataset, num_test_steps=100,is_eval=True)
        test(model, logger, val_dataset, num_test_steps=1000,is_eval=True)
        test(model, logger, test_dataset, num_test_steps=1000)
        pipeline(model, logger, val_dataset.randomized_loader)
        #test_gen(model,val_dataset,logger,num=10)
        #final_eval_generation(model, train_config, logger, verbose=train_config['train_steps']==0)
        return
    
    assert train_config['train_steps']%train_config['eval_every_n_steps']==0
    
    current_best_val_loss = float('inf')
    logger.train_start()
    for mask,x0 in train_dataset:
        model.train()
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        optim.zero_grad()
        loss = model.train_step(mask, x0)
        loss.backward()
        optim.step()
        
        logger.train_step(loss.detach().cpu().item())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps'] == 0:
            model.eval()
            val_loss=test(model, logger, val_dataset, num_test_steps=1000, is_eval=True)
            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                if train_config['save']:
                    logger.log_net(model.cpu(),f"mar_best_{logger.model_name}")
                if model.device.type == 'cuda':
                    model.cuda()
        if logger.step == train_config['train_steps']:
            # load the best checkpoint for test set evaluation
            if train_config['save']:
                name=f"mar_best_{logger.model_name}"
                fp=os.path.join(logger.log_root,f"{name}.pth")
                sd = torch.load(fp, map_location=model.device)
                model.load_state_dict(sd, strict=True)
            model.eval()
            test(model, logger, test_dataset, num_test_steps=1000)
            #test_gen(model,test_dataset,logger,num=10)
            pipeline(model, logger, val_dataset.randomized_loader)
            logger.train_end()
            break
    
    # if train_config['save']:
    #     logger.log_net(model.cpu(),f"mar_{logger.step}_{logger.model_name}")
    return

@torch.no_grad()
def test_gen(model, test_dataset, logger, num=10):
    mask,x0s=next(iter(test_dataset))
    #mask,x0s=mask[:8],x0s[:8]
    # randomly select 8 samples
    idx  = torch.from_numpy(np.random.choice(mask.shape[0], size=4, replace=False)).int()
    try:
        mask, x0s = mask[idx], x0s[idx]
    except IndexError:
        print("WARNING tes_gen train.py")
        idx = idx.to(torch.int64)
        mask, x0s = mask[idx], x0s[idx]
    mask_,x0s_ = model.preprocess(mask,x0s)
    x0s_= x0s_.to(model.device)
    mask_=mask_.to(model.device)
    print("generating")
    steps=[2,8,32]
    for step in steps:
        out=[]
        for _ in range(num):
            res = model.gen(mask_[:, :33],x0s_[:, :33],scope=32,step=step)
            out.append(postprocess_data(model.postprocess(res.cpu()).numpy())) # TODO: handle nan
        logger.test_gen(postprocess_data(x0s.cpu().numpy(), mask.cpu().numpy()), out=out, look_back_len=33*model.seg_size,step=step)
    

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
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        loss = model.train_step(mask, x0)
        acc_loss.append(loss.cpu().item())
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss)
    mode="Eval" if is_eval else "Test"
    info = f"{mode}\n" \
           + f"loss:{acc_loss.mean():.4f}+-{acc_loss.std()/np.sqrt(acc_loss.shape[0]):.4f}\n" 
    print(info)
    logger.log_text(info, "train_log", newline=True)
    if is_eval:
        logger.eval_step(acc_loss.mean())
    return acc_loss.mean()
