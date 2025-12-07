import torch
import numpy as np
from torch import nn
from transformer_modules import TransformerDecoder
from mlp_modules import SimpleMLPAdaLN
from diff_modules import FMDiffuser, EulerSolver

class ARModel(nn.Module):
    """autoregressive time series generation without discrete tokens
    - No patchify
    - inp d = out d 
    """
    def __init__(self,
                 transformer_config: dict,
                 mlp_config:dict,
                 diffusion_config:dict,
                 train_config:dict,
                 device=None, #TODO:...
                 ):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_config = train_config
        self.seg_size = train_config.get('seg_size', 1)
        self.num_fm_per_gd = train_config['num_fm_per_gd']
        self.max_seq_len = train_config['max_seq_len']
        self.build_modules(transformer_config, mlp_config, diffusion_config)
    
    def build_modules(self,transformer_config, mlp_config, diffusion_config):
        self.transformer = TransformerDecoder(**transformer_config).to(self.device)
        self.mlp = SimpleMLPAdaLN(**mlp_config).to(self.device)
        self.diffuser = FMDiffuser(**diffusion_config)
        self.solver = EulerSolver(**diffusion_config)
    
    def calc_loss(self, v_pred, v_gt, mask=None, per_token=False):
        return self.diffuser.calc_loss(v_pred, v_gt, mask, per_token=per_token)
    
    def train_step(self, mask:torch.Tensor, x0:torch.Tensor, per_token_loss=False): # x0 seq = s + 1 TODO: add SOS
        b,s_,d=x0.shape
        s=s_-1
        mask_f32 = mask.clone().to(torch.float32)
        x0_m = torch.cat([x0,mask_f32],dim=-1)
        cond:torch.Tensor = self.get_cond(x0_m[:,:-1])  # [b, s, c]

        # loss = 0
        # for _  in range(self.num_fm_per_gd):
        #     t = self.diffuser.sample_t((b,s), device=self.device)
        #     x, v_gt = self.diffuser.add_noise(x0[:,1:], t)
        #     v_pred = self.pred_v(x, t, cond) # [b, s, d]
        #     loss += self.calc_loss(v_pred, v_gt)
        # return loss / self.num_fm_per_gd

        cond = cond.repeat(1,self.num_fm_per_gd,1)
        t = self.diffuser.sample_t((b,s * self.num_fm_per_gd), device=self.device)
        x0_rep = x0[:,1:].repeat(1,self.num_fm_per_gd,1)
        x, v_gt = self.diffuser.add_noise(x0_rep, t)
        v_pred = self.pred_v(x, t, cond) # [b, s*n, d]
        loss = self.calc_loss(v_pred, v_gt, mask[:,1:].repeat(1,self.num_fm_per_gd,1),per_token=self.num_fm_per_gd if per_token_loss else False)
        return loss
    
    def get_cond(self, x):
        return self.transformer(x)  # [b, s, d]
    
    def pred_v(self, x:torch.Tensor, t:torch.Tensor, cond:torch.Tensor):

        x_ = x.view([-1,x.shape[-1]]).contiguous()
        t_ = t.view(-1).contiguous()
        cond_ = cond.view([-1,cond.shape[-1]]).contiguous()
        
        v_pred = self.mlp(x_, t_, cond_).contiguous()

        return v_pred.view(x.shape).contiguous()
    
    @torch.no_grad()
    def gen(self, mask:torch.Tensor, x: torch.Tensor, scope: int):
        """Autoregressive generative prediction of the future scope tokens
        based on history data (x).
        - No KV cache currently.
        """
        

        if isinstance(scope, int):
            mask_f32 = mask.clone().to(torch.float32)
            x_m = torch.cat([x,mask_f32],dim=-1)
            b, s_h, d=x.shape
            for i in range(scope):
                x_temp = x_m[:,-self.max_seq_len:]
                cond = self.get_cond(x_temp)[:, -1, :]
                ntp = self.solver.generate(self, cond, (b,d)) # [b,d]
                ntp = ntp.view([b,1,d]).contiguous()
                ntp = torch.cat((ntp,torch.zeros_like(ntp)), dim=-1) # [b,1,d*2]
                x_m = torch.cat((x_m,ntp), dim=1)
            return x_m[:,:,:d]
        elif hasattr(scope,"__iter__"):
            mask_f32 = mask[:,:-1].clone().to(torch.float32)
            x_m = torch.cat([x[:,:-1],mask_f32],dim=-1)
            b, s_h, d=x.shape
            s=s_h-1
            ls=[]
            tar,tar_mask=x[:,1:], mask[:,1:]
            tar_mask_ = self.postprocess(tar_mask)[:,:,3:7].bool()
            count=tar_mask_.sum(dim=(0,2),keepdim=False).clamp(min=1)
            x_temp = x_m[:,-self.max_seq_len:]
            cond = self.get_cond(x_temp) #[b,s,d]
            for diff_step in scope:              
                ntp = self.solver.generate(self, cond, (b,s,d),step=diff_step) # [b,s,d]
                temp = (ntp-tar).pow(2)
                temp = self.postprocess(temp)[:,:,3:7]
                temp[tar_mask_]=0
                loss = temp.sum(dim=(0,2),keepdim=False)/count # [s,]
                ls.append(loss.cpu().numpy())
            return np.array(ls)
    
    @torch.no_grad()
    def preprocess(self, mask, x):
        B, S, N = x.shape
        assert S % self.seg_size == 0, "S must be divisible by s"

        x_reshaped = x.reshape(B, S//self.seg_size, self.seg_size, N)
        x_transformed = x_reshaped.permute(0, 1, 3, 2).reshape(B, S//self.seg_size, N*self.seg_size)

        mask_reshaped = mask.reshape(B, S//self.seg_size, self.seg_size, N)
        mask_transformed = mask_reshaped.permute(0, 1, 3, 2).reshape(B, S//self.seg_size, N*self.seg_size)
        return mask_transformed.contiguous() , x_transformed.contiguous()
    
    @torch.no_grad()
    def postprocess(self, x):
        B, S_new, N_new = x.shape
        N = N_new // self.seg_size
        S = S_new * self.seg_size
        
        x_reshaped = x.reshape(B, S_new, N, self.seg_size)
        x_original = x_reshaped.permute(0, 1, 3, 2).reshape(B, S, N)
        return x_original.contiguous()



