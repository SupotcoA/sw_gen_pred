import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from probe_debug import *
from similarity import *
from utils import estimate_mean_and_uncertainty

@torch.no_grad()
def pipeline(model, logger, dataset):
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    #loss_against_sequence_length(model, dataset, logger, num_test_steps=1000)
    for i in range(3):
        diff_loss(model, dataset, logger, num_test_steps=50,metric_idx=i)
    #diff_loss_debug3(model, dataset, logger, num_test_steps=50)
    #diff_loss_debug5(model, dataset, logger, num_test_steps=50)
    #loss_vs_time(model, dataset, logger, num_test_steps=80)

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

def generate_Q_func(z, model, shape, Q_mask, step):
    b,s,d = shape
    xt = torch.randn((b,s,d)).to(model.device)
    step=max(1, step)
    T = torch.linspace(0, 1, step+1)
    for i in reversed(range(1, step+1)):
        t = torch.full((b,s), T[i], device=model.device)
        v_pred = model.pred_v(xt, t, z)
        dt = T[i-1] - T[i]
        xt = flow_step(xt, v_pred, dt)
        xt[Q_mask] = torch.randn_like(xt[Q_mask]) * T[i-1]
    return xt

def calculate_similarity_metric(metric, model, mask, x0, diff_steps,reduce_dim=(0,2)):
    mask_f32 = mask[:,:-1].clone().to(torch.float32)
    x_m = torch.cat([x0[:,:-1],mask_f32],dim=-1)
    b, s_h, d=x0.shape
    s=s_h-1
    ls_m=[]
    ls_s=[]
    cond = model.get_cond(x_m).contiguous() #[b,s,c]
    NUM_SAMPLES_Q_PER_LOOP=16
    for diff_step in diff_steps:      
        kwargs=dict(model=model, shape=(b*NUM_SAMPLES_Q_PER_LOOP,s,d), step=diff_step, Q_mask=mask[:,1:].repeat(NUM_SAMPLES_Q_PER_LOOP,1,1))      
        res_m,res_s=metric(z=cond, x0=x0[:,1:], mask=mask[:,1:], Q_func=generate_Q_func, reduce_dim=reduce_dim, NUM_SAMPLES_Q_PER_LOOP=NUM_SAMPLES_Q_PER_LOOP,**kwargs)
        ls_m.append(res_m.cpu().numpy()) # [s,]
        ls_s.append(res_s.cpu().numpy())
    return np.array(ls_m), np.array(ls_s)



def diff_loss(model, dataset, logger, num_test_steps=50, metric_idx=0):
    print("Evaluating generated distribution similarity metric vs diffusion steps...")
    metric = [cross_entropy_with_kde,mmd,pit_cvm][metric_idx]
    ls_m = []
    ls_s = []
    diff_steps = [4, 8, 16, 32, 64]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        res_m,res_s = calculate_similarity_metric(metric, model, mask, x0, diff_steps,reduce_dim=(0,2)) #[num_diff_steps,s]
        ls_m.append(res_m) #[num_diff_steps,s]
        ls_s.append(res_s) #[num_diff_steps,s]
        if step >= num_test_steps:
            break
    b,s,d=x0.shape
    s-=1
    ls_m,ls_s=estimate_mean_and_uncertainty(m_i=np.array(ls_m),s_i=np.array(ls_s),M=b*d)
    #ls_m [num_diff_steps,s]

    # visualize & save at logger.log_root
    plt.figure(figsize=(10, 6))
    #plt default color cycle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (ds,c) in enumerate(zip(diff_steps,colors)):
        plt.errorbar(np.arange(len(ls_m.shape[0])), ls_m[i], color=c+"80", ecolor=c+"30", yerr=std_loss[i], fmt='-o', capsize=5, label=f'{ds} steps')
        # window_size = 15
        # if len(mean_loss[i]) >= window_size:
        #     # pad the sequence to avoid losing points at the edges
        #     padded_mean = np.pad(mean_loss[i], (window_size//2, window_size//2), mode='edge')
        #     smoothed_mean = np.convolve(padded_mean, np.ones(window_size)/window_size, mode='valid')
        #     plt.plot(np.arange(len(smoothed_mean)), smoothed_mean, color=c, linewidth=3)
    #plt.hlines(y=0.044439464807510376, xmin=0, xmax=len(mean_loss[0])-1, colors='gray', linestyles='dashed')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    metric_name=["Negative Log Likelihood", "Maximum Mean Discrepancy", "Probability Integral Transform"][metric_idx]
    metric_name_s=["NLL","MMD","PIT"][metric_idx]
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, f"{metric_name_s}_vs_seq_step.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    logger.log_arr(ls_m, f"{metric_name_s}_vs_diffsetp_seq_mean")
    logger.log_arr(ls_s, f"{metric_name_s}_vs_diffsetp_seq_std")

    ls_reduced = ls_m.mean(axis=1)
    plt.figure(figsize=(10, 6))
    # loss vs step, averaged over sequence length
    plt.plot(diff_steps, ls_reduced, '-o', linewidth=3)
    plt.yscale('log')
    plt.xlabel('Diffusion Steps')
    plt.ylabel(metric_name)
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, f"{metric_name_s}_vs_diff_steps.png"),
                dpi=300,
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
    




def storm_pred(model, dataset, logger, num_test_steps=250):
    pass
