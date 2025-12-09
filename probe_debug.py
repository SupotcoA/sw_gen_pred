import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def flow_step(xt, v, dt: float):
    return xt + v * dt

def generate(model, cond, shape, mask, step=None, x0=None, t_start=1.0):
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
        xt = flow_step(xt, v_pred, dt)
        xt[mask] = torch.randn_like(xt[mask]) * T[i-1]
    return xt

def generate_eval_intermediate(model, cond, shape, step, x0, mask):
    b = shape[:-1]
    xt = torch.randn(shape).to(model.device)
    mse_x0=[]
    T = torch.linspace(0, 1, step+1)
    for i in reversed(range(1, step+1)):
        t = torch.full(b, T[i], device=model.device)
        v_pred = model.pred_v(xt, t, cond)
        x0_pred = xt - v_pred * t.view(*b, 1)
        mse_x0.append((x0_pred - x0)[~mask].pow(2).mean().cpu().numpy())
        dt = T[i-1] - T[i]
        xt = flow_step(xt, v_pred, dt)
        xt[mask] = torch.randn_like(xt[mask]) * T[i-1]
    return mse_x0

def generate_flowmap(model, cond, shape, steps, mask, feature_idx):
    b = shape[:-1]
    xt0 = torch.randn(shape).to(model.device)
    assert shape[-1]==36, f"{shape}"
    feature_idx_ = [4*idx+2 for idx in feature_idx]
    xts=[[] for _ in steps]
    for j,step in enumerate(steps):
        xt = xt0.clone()
        xts[j].append(xt[:,feature_idx_].cpu().numpy())
        T = torch.linspace(0, 1, step+1)
        for i in reversed(range(1, step+1)):
            t = torch.full(b, T[i], device=model.device)
            v_pred = model.pred_v(xt, t, cond)
            dt = T[i-1] - T[i]
            xt = flow_step(xt, v_pred, dt)
            xt[mask] = torch.randn_like(xt[mask]) * T[i-1]
            xts[j].append(xt[:,feature_idx_].cpu().numpy())
        assert xt.is_contiguous()
        assert torch.allclose(xt[mask],torch.zeros_like(xt[mask])), f"{xt[mask][0:8].cpu()}"
    return xts #[num_diff_steps, diff_step+1, b, num_feature]

def diff_loss_debug5(model, dataset, logger, num_test_steps=50):
    print("Evaluating gen mse vs diffusion steps...")
    num_random_seed = 4
    num_samples = 8
    plot_feature_idx = list(range(9))
    names = "ACE_IMF_Bx ACE_IMF_By ACE_IMF_Bz ACE_Psw ACE_Vsw OMNI_AE OMNI_ASYMH OMNI_PC OMNI_SYMH".split()
    diff_step = [1,4,8,16]
    mask,x0 = next(iter(dataset))
    idx  = torch.from_numpy(np.random.choice(mask.shape[0], size=num_samples, replace=False)).to(torch.int64)
    mask, x0 = mask[idx], x0[idx]
    mask, x0 = model.preprocess(mask,x0)
    mask = mask.to(model.device)
    x0 = x0.to(model.device)
    mask_f32 = mask[:,:-1].clone().to(torch.float32)

    b, s_h, d=x0.shape
    x_m = torch.cat([x0[:,:-1],mask_f32],dim=-1)

    cond = model.get_cond(x_m)[:,-1].contiguous() #[b,c]
    ls_xts=[]
    for _ in range(num_random_seed):          
        xts = generate_flowmap(model, cond, (b,d),steps=diff_step,mask=mask[:,-1],feature_idx=plot_feature_idx)
        ls_xts.append(xts)
    #[num_diff_steps, diff_step+1, b, num_feature]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for ib in range(num_samples):
        fig,axs = plt.subplots(ncols=3,nrows=3,
                                figsize=(12, 12),
                                squeeze=True)
        axs=axs.flatten()
        #plt default color cycle
        
        for ax,fi in zip(axs,range(len(plot_feature_idx))):
            for i, (ds,c) in enumerate(zip(diff_step,colors)):
                for r in range(num_random_seed):
                    xt = [ls_xts[r][i][j][ib,fi] for j in range(len(ls_xts[r][i]))]
                    if r==0:
                        ax.plot(np.linspace(0, 1, ds+1)[::-1], xt, color=c+"80",  marker='o', label=f'{ds} steps')
                    else:
                        ax.plot(np.linspace(0, 1, ds+1)[::-1], xt, color=c+"80",  marker='o')
                ax.scatter([0,0,0,0],x0[ib,-2,fi*4:fi*4+4].cpu().numpy(),marker="+",s=20,c="#00BFFF40")
                ax.scatter([0],[x0[ib,-2,fi*4+3].cpu().item()],marker="*",s=20,c="#00BFFF")
                ax.scatter([0,0,0,0],x0[ib,-1,fi*4:fi*4+4].cpu().numpy(),marker="+",s=20,c="#F8006740")
                ax.scatter([0],[x0[ib,-1,fi*4+2].cpu().item()],marker="*",s=60,c="#F80067")
            ax.set_ylabel(names[fi])
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(logger.log_root, f"flowmap{ib}.png"), 
                    dpi=200, 
                    bbox_inches='tight')
        plt.close()

def diff_loss_debug4(model, dataset, logger, num_test_steps=50):
    print("Evaluating gen mse vs diffusion steps...")
    
    diff_step = [1,4,8,16]
    acc_loss = [[] for _ in range(len(diff_step))]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        mask_f32 = mask[:,:-1].clone().to(torch.float32)

        b, s_h, d=x0.shape
        x_m = torch.cat([x0[:,:-1],mask_f32],dim=-1)
        
        s=s_h-1
        tar,tar_mask=x0[:,1:], mask[:,1:]
        cond = model.get_cond(x_m).contiguous() #[b,s,c]
        for i,diff_step_ in enumerate(diff_step):           
            x0_loss = generate_eval_intermediate(model, cond, (b,s,d),step=diff_step_,x0=tar, mask=tar_mask) # [b,s,d]
            acc_loss[i].append(x0_loss)
        if step >= num_test_steps:
            break
    acc_loss=[np.asarray(loss) for loss in acc_loss] # [ num_diff_steps, num_test_steps, diff_steps]
    mean_loss = [loss.mean(axis=0) for loss in acc_loss]# [num_diff_steps, diff_steps]
    std_loss = [loss.std(axis=0) / np.sqrt(loss.shape[0]) for loss in acc_loss] # [num_diff_steps, diff_steps]

    plt.figure(figsize=(10, 6))
    #plt default color cycle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (ds,c) in enumerate(zip(diff_step,colors)):
        plt.errorbar(np.linspace(0, 1, ds+1)[1:][::-1], mean_loss[i], color=c+"80", ecolor=c+"30", yerr=std_loss[i], fmt='-o', capsize=5, label=f'{ds} steps')
    
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('x0 pred MSE')
    plt.grid(True, linestyle='--', alpha=0.7,which='both')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_root, "x0_pred_mse_vs_t_interm.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()


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
        cond = model.get_cond(x_m).contiguous() #[b,s,c]
        for diff_step_ in diff_step:
            ls2=[]
            for t_start in  t_start_set:            
                ntp = generate(model, cond, (b,s,d),mask=tar_mask,step=diff_step_,x0=x[:,1:],t_start=t_start) # [b,s,d]
                temp = (ntp-tar).pow(2)
                ls2.append(temp[~tar_mask].mean().cpu().numpy())
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