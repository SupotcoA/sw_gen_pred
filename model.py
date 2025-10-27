import torch
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
        self.num_fm_per_gd = train_config['num_fm_per_gd']
        self.max_seq_len = train_config['max_seq_len']
        self.build_modules(transformer_config, mlp_config, diffusion_config)
    
    def build_modules(self,transformer_config, mlp_config, diffusion_config):
        self.transformer = TransformerDecoder(**transformer_config).to(self.device)
        self.mlp = SimpleMLPAdaLN(**mlp_config).to(self.device)
        self.diffuser = FMDiffuser(**diffusion_config)
        self.solver = EulerSolver(**diffusion_config)
    
    def calc_loss(self, v_pred, v_gt):
        return self.diffuser.calc_loss(v_pred, v_gt)
    
    def train_step(self, x0:torch.Tensor): # x0 seq = s + 1 TODO: add SOS
        b,s_,d=x0.shape
        s=s_-1
        cond:torch.Tensor = self.get_cond(x0[:,:-1])  # [b, s, c]

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
        loss = self.calc_loss(v_pred, v_gt)
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
    def gen(self, x: torch.Tensor, scope: int):
        """Autoregressive generative prediction of the future scope tokens
        based on history data (x).
        - No KV cache currently.
        """
        b, s_h, d=x.shape
        for i in range(scope):
            x_temp = x[:,-self.max_seq_len:]
            cond = self.get_cond(x_temp)[:, -1, :]
            ntp = self.solver.generate(self, cond, (b,d)) # [b,d]
            ntp = ntp.view([b,1,d]).contiguous()
            x = torch.cat((x,ntp), dim=1)
        return x
    
    @torch.no_grad()
    def preprocess(self, x):
        return x # TODO
    
    @torch.no_grad()
    def decode(self, x):
        return x # TODO



