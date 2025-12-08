import torch
import numpy as np
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def pipeline(model, logger, dataset):
    model.eval()
    #loss_against_sequence_length(model, dataset, logger, num_test_steps=1000)
    diff_loss(model, dataset, logger, num_test_steps=200)

def loss_against_sequence_length(model, dataset, logger, num_test_steps=1000):
    # how the loss would decrease as we increase the sequence length
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
    acc_loss = []
    diff_step = [2,4,8,16]#,32,64,96]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask,x0)
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        loss = model.gen(mask, x0, scope=diff_step) #[num_diff_steps,s]
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


def diff_loss_debug(model, dataset, logger, num_test_steps=250):
    acc_loss = []
    diff_step = [8,16,32,64,96]
    step = 0
    for mask,x0 in dataset:
        step += 1
        mask, x0 = model.preprocess(mask[:4],x0[:4])
        mask = mask.to(model.device)
        x0 = x0.to(model.device)
        ntp = model.gen(mask, x0, scope=diff_step) #[num_diff_steps,b,s,d]
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
