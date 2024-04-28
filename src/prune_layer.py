import math,time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class PruneConv2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size, stride,prune_ratio=0):
        super(PruneConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size//2
        self.prune_ratio = prune_ratio
        self.weight = nn.Parameter(torch.zeros(out_features,in_features, kernel_size,kernel_size))
        self.mask = torch.ones_like(self.weight.data)
        self.prune_diags ={}
        init.kaiming_uniform_(self.weight)
        self.feature_size = None
        self.rot_ratio = 0
        self.mul  = None
        self.rot = None
        self.he_data = None
        
    # prune in single block
    def prune_weight_single_block(self,weight,mask):
        d = self.he_data[0]
        diag_idx=d-1
        diag_means = []
        prune_idx = []
        while diag_idx>=0: 
            diag = torch.diagonal(weight[:,:,:,:],diag_idx,0,1)
            diag2 = torch.diagonal(weight[:,:,:,:],-d+diag_idx,0,1)
            tmp = torch.cat((diag,diag2),dim=2)
            tmp_dia = torch.mean(torch.abs(tmp)).detach().cpu().numpy()
            diag_means.append(tmp_dia)
            diag_idx-=1
        threshold = np.percentile(diag_means, self.prune_ratio)
        diag_idx=d-1
        while diag_idx>=0: 
            diag = torch.diagonal(weight[:,:,:,:],diag_idx,0,1)
            diag2 = torch.diagonal(weight[:,:,:,:],-d+diag_idx,0,1)
            tmp = torch.cat((diag,diag2),dim=2)
            if torch.mean(torch.abs(tmp))<=threshold:
                pdiag = torch.diagonal(mask[:,:,:,:],diag_idx,0,1)
                pdiag2 = torch.diagonal(mask[:,:,:,:],-d+diag_idx,0,1)
                pdiag.zero_()
                pdiag2.zero_()
                prune_idx.append(diag_idx)
            diag_idx-=1
        return prune_idx    
    
    # prune the weight
    def prune_weight(self):
        self.he_data = self.cal_d()
        d = self.he_data[0]
        K = self.weight.size(0)
        C = self.weight.size(1)
        K = math.floor(K/d)*d
        C = math.floor(C/d)*d
        weight = self.weight[:K,:C,:,:]
        mask = self.mask[:K,:C,:,:]
        print("K//d,C//d",K//d,C//d)
        print("d",d)
        for i in range(K//d):
            for j in range(C//d):
                tmp = self.prune_weight_single_block(weight[i*d:(i+1)*d,j*d:(j+1)*d,:,:],mask[i*d:(i+1)*d,j*d:(j+1)*d,:,:])
                self.prune_diags[(i,j)] = tmp
        current_ratio = 1-torch.sum(self.mask).item()/torch.numel(self.mask)
        print("final_ratio",current_ratio)
        self.rot = self.cal_rot()
        self.prune_ratio = current_ratio 
        self.mul = self.mul * (1-current_ratio)
        
    def forward(self, x):
        if self.feature_size is None:
            assert x.size(2) == x.size(3)
            self.feature_size = x.size(2)
        weight=self.weight*self.mask.to(x.device)
        x = F.conv2d(x,weight,None,self.stride,self.padding)
        return x
    
    def next_power_2(delf,d):
        p = math.ceil(math.log2(d))
        return int(pow(2,p))

    # cal_rot and comm are used to compute latency of each layer, each block size
    def cal_d(self):
        n = 8192
        m = self.feature_size ** 2
        d1 = self.in_features
        d2 = self.out_features
        b=1
        min_rot = 1e8
        d_min = int(min(d2/b,d1/b))
        final_mp = 0
        final_d = 0
        final_ri = 0
        final_ro = 0
        for ri in range(1,(d_min)+1):
            for ro in range(1,(d_min)+1):
                d=int(ri*ro)
                m_p=int(n/b/d)
                if m*d_min*b<n:
                    if d!=d_min:
                        continue
                    i = 1
                    while i<=m:
                        next_pow_2 = self.next_power_2(i*b)
                        if next_pow_2*d>n:
                            break
                        i+=1
                    m_p=i-1
                if d>d_min or m_p>m:
                    continue
                if b!=1:
                    next_pow_2 = self.next_power_2(m_p*b)
                    if next_pow_2*d>n:
                        continue
                tmp=m*d1*(ri-1)/(m_p*b*d)+m*d2*(ro-1)/(m_p*b*d)
                if tmp<min_rot:
                    min_rot=tmp
                    final_d=d
                    final_mp = m_p
                    final_ri = ri
                    final_ro = ro
                    # print("ri,ro,d,m_p",ri,ro,d,m_p)
        # print(min_rot)
        # print("n,m,d1,d2,b:",n,m,d1,d2,b)
        # print("final_mp,final_d:",final_mp,final_d)
        mul = math.ceil(1.0*m/final_mp)*math.ceil(1.0*d1/b/final_d)*math.ceil(1.0*d2/b/final_d)*final_d
        self.mul = mul
        return [final_d,final_ri,final_ro,min_rot]
    
    def cal_rot(self):
        d,ri,ro,min_rot = self.he_data
        print("ri,ro:",ri,ro)
        K = self.weight.size(0)
        C = self.weight.size(1)
        K = math.floor(K/d)*d
        C = math.floor(C/d)*d
        reduce_num_rot = 0
        '''
            单个小方格内,ri剪枝条件: 所有的ro组内,相同位置的W都是0,那么就可以剪枝小方格内的对应ri
            单个小方格内,ro剪枝条件: 单个ro组内,所有w都是0,就可以剪枝
            对于整体W的情况,如果剪枝要使得ri减少,那么对于W的一整行,也就是K维度,所有的小方格内都必须满足ri剪枝条件
            对于整体W的情况,如果剪枝要使得ro减少,那么对于W的一整列,也就是C维度,所有的小方格内都必须满足ro剪枝条件
        '''
        if ro>=2:
            for i in range(K//d):
                for k in range(ro-1):
                    can_reduce = True
                    for j in range(C//d):
                        idx = self.prune_diags[(i,j)]
                        for l in range(ri):
                            if k*ri+l not in idx:
                                can_reduce = False
                                break
                        if not can_reduce:
                            break
                    if can_reduce:
                        reduce_num_rot+=1
        if ri>=2:
            for j in range(C//d):
                for k in range(ri-1):
                    can_reduce = True
                    for i in range(K//d):
                        idx = self.prune_diags[(i,j)]
                        for l in range(ro):
                            if l*ri+k not in idx:
                                can_reduce = False
                                break
                        if not can_reduce:
                            break
                    if can_reduce:
                        reduce_num_rot+=1
        print("min_rot",min_rot)
        self.rot_ratio = (min_rot-reduce_num_rot)/min_rot
        min_rot-=reduce_num_rot
        print("min_rot",min_rot)
        return min_rot
                
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, kernel_size={self.kernel_size}, prune_ratio={self.prune_ratio},rot_ratio={self.rot_ratio}'

# for test
if __name__ == '__main__':
    conv = PruneConv2d(96, 96, 1, 1,prune_ratio=88,feature_size=32)
    x = torch.randn(1,96,32,32)
    y = conv(x)
    