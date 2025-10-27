import torch
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def calculate_num_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def tensor2bgr(tensor):
    imgs = torch.clip(torch.permute(tensor, [0, 2, 3, 1]).cpu().add(1).mul(127.5), 0, 255)
    return imgs.numpy().astype(np.uint8)[:, :, :, ::-1]

class Logger:
    def __init__(self,
                 log_every_n_steps=100,
                 log_root=None,
                 model_name=None
                 ):
        self.log_root = log_root
        self.model_name=model_name
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.step = 0
        self.skip_step = 0
        self.log_every_n_steps = log_every_n_steps
        self.time = 0
        self.loss_accum = 0
        self.train_loss = []
        self.train_memory = []
        
    def train_start(self):
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.time = time.time()
        torch.cuda.reset_peak_memory_stats()
    
    def train_resume(self):
        self.time = time.time()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def skip_train_step(self, default_loss):
        self.step += 1
        self.skip_step += 1
        self.loss_accum += default_loss

    @torch.no_grad()
    def train_step(self, loss):
        self.step += 1
        self.loss_accum += loss
        self.train_loss.append(loss)
        
        # Record peak memory for this step (in GB)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.train_memory.append(peak_mem)
        torch.cuda.reset_peak_memory_stats()
        
        if self.step % self.log_every_n_steps == 0:
            dt = time.time() - self.time
            current_peak_mem = max(self.train_memory[-self.log_every_n_steps:])
            info = (f"Train step {self.step}\n"
                   f"loss: {self.loss_accum/self.log_every_n_steps:.4f}\n"
                   f"time per kstep: {dt/self.log_every_n_steps*1000:.0f}\n"
                   f"peak GPU mem: {current_peak_mem:.1f} GB\n")
            
            print(info)
            self.log_text(info, "train_log")
            self.time = time.time()
            self.loss_accum = 0
    
    def test_gen(self, x0s, out:list, idx):
        x0s = x0s.copy()
        plt.style.use('seaborn')
        t = np.arange(x0s.shape[1])
        t_res = np.arange(idx+1, x0s.shape[1])
        for b in range(x0s.shape[0]):
            fig,axs = plt.subplots(ncols=1,nrows=3,sharex=True,figsize=(16, 12),squeeze=True)

            plot_dim = [0, 4, 7]

            for dim, ax in zip(plot_dim,axs):
                ax.plot(t, x0s[b, :, dim], linewidth=5, color='#2E86C1')
                for res in out:
                    ax.plot(t_res, res[b, idx:, dim], linewidth=5, color="#EB38CA",alpha=1/len(out))
            
            # Add tight layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_root, f"gen{b}.png"), 
                        dpi=100, 
                        bbox_inches='tight')
            plt.close()
    
    def train_end(self):
        self.log_text(f"Skipped steps: {self.skip_step}","train_log")
        # Create figure with improved style
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(10, 6))
        
        # Plot learning curve with better styling
        plt.plot(self.train_loss, linewidth=2, color='#2E86C1', alpha=0.8)
        
        # Add rolling average for smoother visualization
        window_size = min(1000, len(self.train_loss))
        rolling_mean = np.convolve(self.train_loss, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
        plt.plot(range(window_size-1, len(self.train_loss)), 
                 rolling_mean, 
                 linewidth=2.5, 
                 color='#E74C3C', 
                 label='Rolling average')
        
        # Customize appearance
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, pad=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add tight layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_root, "train_stats.png"), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.close()
    
    def log_text(self, text, fname="log", newline=True):
        if newline:
            text = "\n"+text
        path = os.path.join(self.log_root, f"{fname}.txt")
        with open(path, 'a') as f:
            f.write(text)

    def generation_start(self):
        self.eval_time = time.time()
        torch.cuda.reset_peak_memory_stats()

    def generation_end(self):
        dt = time.time() - self.eval_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        info = (f"Generation time: {dt:.0f}s\n"
                f"Peak GPU memory: {peak_mem:.1f}GB\n")
        
        print(info)
        self.log_text(info, "train_log")

    @torch.no_grad()
    def log_images(self,imgs,nrow,ncol,fname):
        # imgs: torch.Tensor shape(B=nrow*ncol, C, H, W)
        imgs = tensor2bgr(imgs)
        h, w, c = imgs.shape[1:]
        base = np.zeros((h * nrow, w * ncol, c), dtype=np.uint8)
        for i in range(nrow):
            for j in range(ncol):
                base[i * h:i * h + h, j * w:j * w + w, :] = imgs[j * nrow + i]
        fp = os.path.join(self.log_root, f"{fname}")
        if os.path.exists(fp+".png"):
            num=1
            while os.path.exists(fp+f"({num}).png"):
                num+=1
            fp=fp+f"({num})"
        fp+=".png"
        
        cv2.imwrite(fp, base)
    
    def log_net(self,net,name):
        torch.save(net.state_dict(),os.path.join(self.log_root,f"{name}.pth"))

@torch.no_grad()
def check_ae(model,x0, batch_size=9):
    # check if the decoder is working
    imgs=model.decode(x0[:batch_size], need_postprocess=False)
    return imgs
