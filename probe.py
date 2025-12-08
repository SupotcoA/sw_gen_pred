import torch
import numpy as np
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def pipeline(model, logger, dataset):
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    #loss_against_sequence_length(model, dataset, logger, num_test_steps=1000)
    diff_loss(model, dataset, logger, num_test_steps=50)
    loss_vs_time(model, dataset, logger, num_test_steps=80)

    peak_memory=torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak memory usage during probing: {peak_memory:.2f} GB")

def loss_against_sequence_length(model, dataset, logger, num_test_steps=1000):
    # how the loss would decrease as we increase the sequence length
    print("Evaluating loss vs sequence length...")
    acc_loss = []
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        loss = model.train_step(mask, x0, per_token_loss=True) # [s,]
        acc_loss.append(loss.cpu().numpy())
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss)
    mean_loss = acc_loss.mean(axis=0)
    std_loss = acc_loss.std(axis=0) / np.sqrt(acc_loss.shape[0])
    # visualize & save at logger.log_root
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(np.arange(len(mean_loss)), mean_loss, yerr=std_loss, fmt='-o', ecolor='#00BFFF80', capsize=5)
    # also plot a smoothed version
    window_size = 15
    if len(mean_loss) >= window_size:
        # pad the sequence to avoid losing points at the edges
        padded_mean = np.pad(mean_loss, (window_size//2, window_size//2), mode='edge')
        smoothed_mean = np.convolve(padded_mean, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(len(smoothed_mean)), smoothed_mean, color="#F80067", label='Smoothed', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    #plt.legend()
    
    # Add tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, "loss_vs_seq.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def diff_loss(model, dataset, logger, num_test_steps=250):
    print("Evaluating gen mse vs diffusion steps...")
    acc_loss = []
    diff_step = [1,4,16,64]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        loss = model.gen(mask, x0, step=diff_step) #[num_diff_steps,s]
        acc_loss.append(loss)
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss) # [num_test_steps, num_diff_steps, s]
    mean_loss = acc_loss.mean(axis=0) # [num_diff_steps,s]
    std_loss = acc_loss.std(axis=0) / np.sqrt(acc_loss.shape[0]) # [num_diff_steps,s]
    # visualize & save at logger.log_root
    plt.figure(figsize=(10, 6))
    #plt default color cycle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (ds,c) in enumerate(zip(diff_step,colors)):
        plt.errorbar(np.arange(len(mean_loss[i])), mean_loss[i], color=c+"80", ecolor=c+"30", yerr=std_loss[i], fmt='-o', capsize=5, label=f'{ds} steps')
        window_size = 15
        if len(mean_loss[i]) >= window_size:
            # pad the sequence to avoid losing points at the edges
            padded_mean = np.pad(mean_loss[i], (window_size//2, window_size//2), mode='edge')
            smoothed_mean = np.convolve(padded_mean, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(len(smoothed_mean)), smoothed_mean, color=c, linewidth=3)
    #plt.hlines(y=0.044439464807510376, xmin=0, xmax=len(mean_loss[0])-1, colors='gray', linestyles='dashed')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, "mse_vs_seq_step.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    # loss vs step, averaged over sequence length
    plt.plot(diff_step, mean_loss.mean(axis=1), '-o', linewidth=3)
    plt.yscale('log')
    plt.xlabel('Diffusion Steps')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, "mse_vs_diff_steps.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()

def generate(model, cond, shape, step=None, x0=None, t_start=1.0):
    b = shape[:-1]
    xt = torch.randn(shape).to(model.device)
    if x0 is not None:
        xt = xt * t_start + x0 * (1 - t_start)
    else:
        assert t_start == 1.0, "x0 must be provided if t_start < 1.0"
    step = step if step is not None else 32
    step=max(1, int(step*t_start))
    T = torch.linspace(0, t_start, step+1)
    for i in reversed(range(1, step+1)):
        t = torch.full(b, T[i], device=model.device)
        v_pred = model.pred_v(xt, t, cond)
        dt = T[i-1] - T[i]
        xt = step(xt, v_pred, dt)
    return xt

def diff_loss_debug3(model, dataset, logger, num_test_steps=50):
    print("Evaluating gen mse vs diffusion steps...")
    acc_loss = []
    diff_step = [1,4,8,16]
    t_start_set = [1.0, 0.75, 0.5, 0.25]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        mask_f32 = mask[:,:-1].clone().to(torch.float32)
        x=x0
        b, s_h, d=x.shape
        x_m = torch.cat([x[:,:-1],mask_f32],dim=-1)
        
        s=s_h-1
        ls=[]
        tar,tar_mask=x[:,1:], mask[:,1:]
        tar_mask_ = model.postprocess(tar_mask).bool() # [b,s*4,4]
        cond = model.get_cond(x_m).contiguous() #[b,s,c]
        for diff_step in step:
            ls2=[]
            for t_start in  t_start_set:            
                ntp = generate(model, cond, (b,s,d),step=diff_step,x0=x[:,1:],t_start=t_start) # [b,s,d]
                temp = (ntp-tar).pow(2)
                temp = model.postprocess(temp)
                ls2.append(temp[tar_mask_].mean().cpu().numpy())
            ls.append(ls2)
        acc_loss.append(ls)
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss) # [num_test_steps, num_diff_steps, num_t_start]
    mean_loss = acc_loss.mean(axis=0) # [num_diff_steps, num_t_start]
    std_loss = acc_loss.std(axis=0) / np.sqrt(acc_loss.shape[0]) # [num_diff_steps, num_t_start]

    plt.figure(figsize=(10, 6))
    #plt default color cycle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (ds,c) in enumerate(zip(diff_step,colors)):
        plt.errorbar(t_start_set, mean_loss[i], color=c+"80", ecolor=c+"30", yerr=std_loss[i], fmt='-o', capsize=5, label=f'{ds} steps')
    
    plt.yscale('log')
    plt.xlabel('t_start')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, "mse_vs_t_start.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def diff_loss_debug2(model, dataset, logger, num_test_steps=50):
    print("Evaluating gen mse vs diffusion steps...")
    acc_loss = []
    diff_step = [1,4,16,64]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        loss = model.gen(mask, x0, scope="debug",step=diff_step) #[num_diff_steps,]
        acc_loss.append(loss)
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss) # [num_test_steps, num_diff_steps]
    mean_loss = acc_loss.mean(axis=0) # [num_diff_steps,]
    std_loss = acc_loss.std(axis=0) / np.sqrt(acc_loss.shape[0]) # [num_diff_steps]

    plt.figure(figsize=(10, 6))
    # loss vs step, averaged over sequence length
    plt.plot(diff_step, mean_loss, '-o', linewidth=3)
    plt.yscale('log')
    plt.xlabel('Diffusion Steps')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, "mse_vs_diff_steps.png"),
                dpi=100,
                bbox_inches='tight')
    plt.close()

def loss_vs_time(model, dataset, logger, num_test_steps=250):
    print("Evaluating loss vs diffusion time t...")
    ts=torch.linspace(0,1,10)
    acc_loss = []
    acc_loss_x0=[]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        b,s_,d=x0.shape
        s=s_-1
        mask_f32 = mask.clone().to(torch.float32)
        x0_m = torch.cat([x0,mask_f32],dim=-1)
        cond:torch.Tensor = model.get_cond(x0_m[:,:-1])  # [b, s, c]
        ls=[]
        lx=[]
        for i in range(len(ts)):
            t = torch.full((b,s), ts[i], device=model.device)
            x, v_gt = model.diffuser.add_noise(x0[:,1:], t)
            v_pred = model.pred_v(x, t, cond) # [b, s, d]
            x0_pred = x - v_pred * t.view(b, s, 1)
            loss = model.calc_loss(v_pred, v_gt, mask[:,1:], per_token=False)
            ls.append(loss.cpu().item())
            loss_x0 = (x0_pred - x0[:,1:])[~mask[:,1:].bool()].pow(2).mean().cpu().item()
            lx.append(loss_x0)
        acc_loss.append(ls)
        acc_loss_x0.append(lx)
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss) # [num_test_steps, num_time_points]
    mean_loss = acc_loss.mean(axis=0) # [num_time_points]
    std_loss = acc_loss.std(axis=0) / np.sqrt(acc_loss.shape[0]) # [num_time_points]
    acc_loss_x0=np.asarray(acc_loss_x0) # [num_test_steps, num_time_points]
    mean_loss_x0 = acc_loss_x0.mean(axis=0) # [num_time_points]
    std_loss_x0 = acc_loss_x0.std(axis=0) / np.sqrt(acc_loss_x0.shape[0]) # [num_time_points]

    # visualize & save at logger.log_root
    fig,ax=plt.subplots(figsize=(10, 6))
    ax.errorbar(ts.numpy(), mean_loss, yerr=std_loss, fmt='-o',color="#00BFFF", ecolor='#00BFFF40', capsize=5)
    ax.set_xlabel('Diffusion Time t')
    ax.set_ylabel('loss')
    ax.grid(True, linestyle='--', alpha=0.7,axis='x')

    ax2=ax.twinx()
    ax2.errorbar(ts.numpy(), mean_loss_x0, yerr=std_loss_x0, fmt='-o', color='#F80067', ecolor='#F8006740', capsize=5)
    ax2.set_ylabel('x0 loss', color='#F80067')
    ax2.tick_params(axis='y', labelcolor='#F80067')

    plt.tight_layout()
    fig.savefig(os.path.join(logger.log_root, f"loss_vs_t.png"), 
                dpi=100, transparent=False, 
                bbox_inches='tight')
    plt.close()
    


def diff_loss_debug(model, dataset, logger, num_test_steps=250):
    acc_loss = []
    diff_step = [8,16,32,64,96]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask[:4],x0[:4])
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        ntp = model.gen(mask, x0, step=diff_step) #[num_diff_steps,b,s,d]
        acc_loss=ntp #[]
        break
    acc_loss=np.asarray(acc_loss) #[num_diff_steps,b,s,d]
    
    x0s = model.postprocess(torch.from_numpy(acc_loss[-1])).numpy() #[b,s*4,9]
    tar = model.postprocess(x0[:,1:]).cpu().numpy() #[b,s*4,9]
    t = np.arange(x0s.shape[1])
    for b in range(x0s.shape[0]):
        plot_dim = [0, 2, 3, 6, 7]
        names = ['Bx', 'Bz', 'AE', 'SYM-H', 'P']
        fig,axs = plt.subplots(ncols=1,nrows=len(plot_dim),
                                sharex=True,
                                figsize=(14, 4*len(plot_dim)),
                                squeeze=True)

        for dim,name, ax in zip(plot_dim,names,axs):
            ax.plot(t, tar[b, :, dim], linewidth=5, color='#00BFFF')
            ax.plot(t, x0s[b, :, dim], linewidth=5, color="#F80067")
            ax.set_ylabel(name, fontsize=16)
        # Add tight layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(logger.log_root, f"single_step_gen{b}.png"), 
                    dpi=100, 
                    bbox_inches='tight')
        plt.close()

def storm_pred(model, dataset, logger, num_test_steps=250):
    pass
