
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 dim=1024,
                 num_heads=1,
                 norm_layer=nn.LayerNorm,
                 num_group_token=4,
                 num_output_group=4,
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.,
                 dtype=torch.float16):
        
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_group_token=num_group_token
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        self.group_proj = nn.Linear(dim,dim)
        self.norm_post_tokens = norm_layer(dim)
        groupToken = torch.empty(self.num_group_token, self.dim, dtype=dtype)
        nn.init.normal_(groupToken, std=0.02)
        self.groupToken = nn.Parameter(groupToken)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype=dtype
        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.post_proj = nn.Linear(num_group_token,num_output_group)

    def init_group(self,parameter):
        #pass
        self.groupToken = nn.Parameter(parameter)

    def forward(self, x, clsToken):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        groupToken=torch.unsqueeze(self.groupToken, dim=0)
        group_tokens=groupToken.repeat(x.size(0), 1, 1)
        
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        group_tokens = self.group_proj(group_tokens)
        group_tokens = self.pre_assign_attn(group_tokens, x)
        group_tokens=self.post_proj(group_tokens.transpose(1, 2)).transpose(1, 2)
        group_tokens,_ = self.assign(group_tokens, x, return_attn=False)
       
        return group_tokens

       
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x




class QueryBlock(nn.Module):
    def __init__(self,
                dim=512,
                num_token=16,
                dtype=torch.float,
                low_dim=80
                ):
        super(QueryBlock,self).__init__()
        self.dim=dim
        self.num_token=num_token
        self.dtype=dtype
        token = torch.empty(self.num_token, self.dim, dtype=dtype)
        nn.init.normal_(token, std=0.02)
        self.token= nn.Parameter(token)

        self.spatial = nn.Parameter(torch.tensor(0.5, dtype=dtype)  )
        self.scale1 = nn.Parameter(torch.tensor(3.0, dtype=dtype)  )
        self.scale2 = nn.Parameter(torch.tensor(3.0, dtype=dtype)  )
        self.k=nn.Linear(low_dim,low_dim)
        self.q=nn.Linear(low_dim,low_dim)
        self.v=nn.Linear(low_dim,low_dim)
        self.proj=nn.Linear(dim,dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.eye_(self.k.weight)
        nn.init.zeros_(self.k.bias)
        nn.init.eye_(self.q.weight)
        nn.init.zeros_(self.q.bias)
        nn.init.eye_(self.v.weight)
        nn.init.zeros_(self.v.bias)

        self.down0=nn.Linear(dim,low_dim)
        self.down1=nn.Linear(dim,low_dim)
        self.up=nn.Linear(low_dim,dim)

        self.cross=CrossAttnBlock(dim=low_dim,num_heads=1,mlp_ratio=2,drop_path=0.1)
    def init_token(self,token):
        self.token = nn.Parameter(token)
        self.down0.weight =  nn.Parameter(token)
        self.down1.weight =  nn.Parameter(token)

    #原本的方法
    def forward(self,x,mask):
        x=self.proj(x)@self.token.t()
        if mask is not None:
            x = x+mask[:,:,None]     
        
        x_attn=torch.nn.functional.softmax(x*50,dim=1)
        return x_attn

    def low_dim(self,feat,features,mask=None):
        features_low=self.down0(features)
        feat_low=self.down1(feat)
        features_low=features_low.unsqueeze(1)
        features_low=self.cross(features_low,feat_low,mask=None)
        return features_low.squeeze()





class Net(nn.Module):
    def __init__(self, dim,proj=None):
        super().__init__()
        token = torch.empty(1, dim)
        nn.init.normal_(token, std=0.02)
        self.token= nn.Parameter(token)
        self.cross=CrossAttnBlock(dim,2)
        
    def forward(self, x, x_2, x_3):
        H=math.floor(math.sqrt(x.shape[1]))
        x=x.reshape(x.shape[0],H,H,x.shape[2])
        token=self.token.repeat(x.size(0), 1, 1)
        x2=self.split_tensor(x,2)
        x3=self.split_tensor(x,3)
        loss=0
        for i,x0 in enumerate(x2):
            x0=self.cross(token, x0.reshape(x.shape[0],-1,x.shape[3]) )
            loss+=self.cos_loss(x0,x_2[:,i,:])
        for i,x0 in enumerate(x3):
            x0=self.cross(token, x0.reshape(x.shape[0],-1,x.shape[3]) )
            loss+=self.cos_loss(x0,x_3[:,i,:])
        return loss
    

    def cos_loss(self,t1,t2):
        cos_sim = F.cosine_similarity(t1, t2, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss
    
    def split_tensor(self,tensor,num_split):
        N, H, W, C=tensor.shape
        # 计算切片大小
        slice_height = math.ceil(H / num_split)
        slice_width = math.ceil(W / num_split)

        # 计算步长
        step_height = (H - slice_height) // 2
        step_width = (W - slice_width) // 2
        splits = []
        for i in range(num_split):
            for j in range(num_split):
                start_height = i * step_height
                end_height = start_height + slice_height
                start_width = j * step_width
                end_width = start_width + slice_width

                # 调整边界条件
                start_height = max(start_height, 0)
                end_height = min(end_height, H)
                start_width = max(start_width, 0)
                end_width = min(end_width, W)

                split = tensor[:, start_height:end_height, start_width:end_width, :]
                splits.append(split)
        return splits
    def interface(self, x):
        H=math.floor(math.sqrt(x.shape[1]))
        x=x.reshape(x.shape[0],H,H,x.shape[2])
        token=self.token.repeat(x.size(0), 1, 1)
        x2=self.split_tensor(x,2)
        x3=self.split_tensor(x,3)
        x_2=[]
        x_3=[]
        for i,x0 in enumerate(x2):
            x0=self.cross(token, x0.reshape(x.shape[0],-1,x.shape[3]) )
            x_2.append(x0)
        for i,x0 in enumerate(x3):
            x0=self.cross(token, x0.reshape(x.shape[0],-1,x.shape[3]) )
            x_3.append(x0)
        return  x_2, x_3