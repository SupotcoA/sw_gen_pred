import torch
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def calculate_num_params(model, trainable_only=False):
    if hasattr(model, 'parameters'):
        model = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in model if p.requires_grad)
    return sum(p.numel() for p in model)

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
        self.eval_loss_history = []
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
    
    @torch.no_grad()
    def eval_step(self, loss):
        self.eval_loss_history.append((self.step, loss))
    
    @torch.no_grad()
    def test_gen(self, x0s, out:list, look_back_len,step=None):
        self.log_arr(x0s,"test_gen_x0s")
        self.log_arr(np.asarray(out),f"test_gen_out_{look_back_len}")
        x0s = x0s.copy()
        t = np.arange(x0s.shape[1])
        t_res = np.arange(look_back_len, x0s.shape[1])
        for b in range(x0s.shape[0]):
            plot_dim = [0, 2, 3, 5]
            names = ['Bx', 'Bz', 'AE', 'SYM-H']
            fig,axs = plt.subplots(ncols=1,nrows=len(plot_dim),
                                   sharex=True,
                                   figsize=(14, 4*len(plot_dim)),
                                   squeeze=True)

            for dim,name, ax in zip(plot_dim,names,axs):
                ax.plot(t, x0s[b, :, dim], linewidth=5, color="#009CD0")
                for res in out:
                    ax.plot(t_res, res[b, look_back_len:, dim], linewidth=5, color="#FF006A",alpha=min(3/len(out),0.99))
                ax.set_ylabel(name, fontsize=16)
            # Add tight layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_root, f"gen{b}_s{step}.png"), 
                        dpi=100, 
                        bbox_inches='tight')
            plt.close()
    
    def train_end(self):
        self.log_text(f"Skipped steps: {self.skip_step}","train_log")
        # Create figure with improved style
        #plt.style.use('seaborn')
        fig = plt.figure(figsize=(10, 6))
        
        # Plot learning curve with better styling
        plt.plot(self.train_loss, linewidth=1, color="#FF009930")
        self.log_arr(np.asarray(self.train_loss),"train_loss")
        x=[step for step, loss in self.eval_loss_history]
        y=[loss for step, loss in self.eval_loss_history]
        self.log_arr(np.asarray([x,y]),"eval_step_eval_loss")
        plt.plot(x,y, 'o-', color="#00BFFF", label='Eval Loss',linewidth=3)
        
        # Add rolling average for smoother visualization
        window_size = min(1000, len(self.train_loss))
        rolling_mean = np.convolve(self.train_loss, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
        plt.plot(range(window_size//2-1, window_size//2-1+rolling_mean.shape[0]), 
                 rolling_mean, 
                 linewidth=3, 
                 color="#FF0067", 
                 label='Rolling average')
        
        # Customize appearance
        plt.yscale('log')
        plt.ylim(top=0.2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, pad=15)
        plt.grid(True, linestyle='--', alpha=0.7,which='both')
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
    
    def log_arr(self,arr:np.ndarray,name):
        np.save(os.path.join(self.log_root,f"{name}.npy"),arr)

@torch.no_grad()
def check_ae(model,x0, batch_size=9):
    # check if the decoder is working
    imgs=model.decode(x0[:batch_size], need_postprocess=False)
    return imgs

@torch.no_grad()
def estimate_mean_and_uncertainty(m_i, s_i, M):
    """
    根据分组统计量计算总体均值和不确定度
    
    参数:
    m_i: numpy数组, 形状为[N, ...], 每组数据的均值
    s_i: numpy数组, 形状为[N, ...], 每组数据的标准差(假设为ddof=1的无偏估计)
    M: int, 每组数据的样本数
    
    返回:
    mean_estimate: numpy数组, 形状为[...], 总体均值的估计
    uncertainty: numpy数组, 形状为[...], 总体均值的标准误差
    """
    # 计算组数N
    N = m_i.shape[0]
    
    # 1. 计算总体均值(各组均值的平均)
    mean_estimate = np.mean(m_i, axis=0)
    
    # 2. 计算组间平方和
    # diff = m_i - mean_estimate (广播)
    diff = m_i - mean_estimate[np.newaxis, ...]
    SS_between = M * np.sum(diff**2, axis=0)
    
    # 3. 计算组内平方和
    # 假设s_i是ddof=1的无偏估计, 所以组内方差为s_i^2
    # 组内离差平方和 = (M-1) * s_i^2
    SS_within = np.sum((M - 1) * s_i**2, axis=0)
    
    # 4. 计算总平方和
    SS_total = SS_between + SS_within
    
    # 5. 计算总体方差的估计
    # 总自由度 = N*M - 1
    total_dof = N * M - 1
    variance_estimate = SS_total / total_dof
    
    # 6. 计算总体均值的标准误差(不确定度)
    uncertainty = np.sqrt(variance_estimate / (N * M))
    
    return mean_estimate, uncertainty