import torch
from torch import nn

class FMDiffuser:
    """xt = (1 - t) * x0 + t * n"""
    def __init__(self, **kwargs):
        pass

    @torch.no_grad()
    def sample_t(self, shape, device=None):
        # shape b, s
        # SD3 lognorm(0,1)
        t = torch.randn(shape,device=device)
        t = torch.sigmoid(t)
        return t


    @torch.no_grad()
    def add_noise(self, x0:torch.Tensor, t:torch.Tensor):
        # x0 [B, S, D]
        # t [B, S]
        b,s,d=x0.shape
        t_ = t.view([b, s, 1])
        n = torch.randn_like(x0)
        v_gt = n - x0
        return (1 - t_) * x0 + t_ * n, v_gt
    
    def calc_loss(self, v_pred, v_gt, mask, t=None, per_token=False):
        if per_token:
            temp = postprocess((v_pred - v_gt).pow(2))[:,:,3:7]
            mask = postprocess(mask)[:,:,3:7].bool()
            b, s, d = temp.shape
            temp = temp.contiguous().view(b,per_token,s//per_token,d)
            temp_mask = mask.contiguous().view(b,per_token,s//per_token,d)
            temp[temp_mask] = 0
            count = (~temp_mask).sum(dim=(0,1,3), keepdim=False)
            temp = temp.sum(dim=(0,1,3), keepdim=False) / count.clamp(min=1)
            return temp
        return (v_pred - v_gt)[~mask].pow(2).mean()
    
def postprocess(x):
    B, S_new, N_new = x.shape
    N = N_new // 4
    S = S_new * 4
    
    x_reshaped = x.reshape(B, S_new, N, 4)
    x_original = x_reshaped.permute(0, 1, 3, 2).reshape(B, S, N)
    return x_original.contiguous()
    
    

class EulerSolver:
    """SD3 used Euler solver"""
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.t = torch.linspace(0, 1, num_steps)

    def get_t(self, i):
        # i: step index
        return self.t[i]
    
    @torch.no_grad()
    def step(self, xt, v, dt: float):
          return xt + v * dt  # dt < 0
    
    @torch.no_grad()
    def generate(self, model, cond, shape,step=None):
        b = shape[:-1]
        xt = torch.randn(shape).to(model.device)
        step = step if step is not None else self.num_steps
        for i in reversed(range(1, step)):
            t = torch.full(b, self.t[i], device=model.device)
            v_pred = model.pred_v(xt, t, cond)
            dt = self.t[i-1] - self.t[i]
            xt = self.step(xt, v_pred, dt)
        return xt  # [b, d]