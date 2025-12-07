import torch
import numpy as np
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def pipeline(model, logger, dataset):
    model.eval()
    loss_against_sequence_length(model, dataset, logger, num_test_steps=1000)
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
    window_size = 13
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
    diff_step = [8,16,32,64,96]
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
    
    for i, (ds,c) in enumerate(diff_step,colors):
        plt.errorbar(np.arange(len(mean_loss[i])), mean_loss[i], color=c+"A0", ecolor=c+"50", yerr=std_loss[i], fmt='-o', capsize=5, label=f'{ds} steps')
        window_size = 13
        if len(mean_loss) >= window_size:
            # pad the sequence to avoid losing points at the edges
            padded_mean = np.pad(mean_loss, (window_size//2, window_size//2), mode='edge')
            smoothed_mean = np.convolve(padded_mean, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(len(smoothed_mean)), smoothed_mean, color=c, label='Smoothed', linewidth=4)
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
