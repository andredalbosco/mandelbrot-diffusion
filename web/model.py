"""Inference-only tinygrad port of the repository's conditional JiT."""
import numpy as np
from tinygrad import Tensor

class JiT:
    def __init__(self, config, state):
        self.config=config
        self.weights={k:Tensor(v.detach().float().cpu().numpy()).realize() for k,v in state.items()}
        self.frequencies=Tensor(np.exp(-np.log(10000)*np.arange(128,dtype=np.float32)/128).astype(np.float32)).realize()

    def linear(self,x,name):
        return x.linear(self.weights[name+'.weight'].T,self.weights.get(name+'.bias'))

    def norm(self,x,name):
        return x * (x.square().mean(axis=-1,keepdim=True)+1e-6).rsqrt() * self.weights[name+'.weight']

    def forward(self,x,t,evidence,mask):
        c=self.config; p=c['patch_size'];d=c['hidden_size'];h=c['num_heads'];g=128//p
        w=self.weights
        inp=x.cat((mask>0).where(evidence,0),mask,dim=1)
        tokens=inp.conv2d(w['patch_embed.0.weight'],stride=p).conv2d(w['patch_embed.1.weight'],w['patch_embed.1.bias'])
        tokens=tokens.reshape(1,d,g*g).permute(0,2,1)+w['positions']
        angles=t.reshape(1,1)*self.frequencies.reshape(1,128)
        time=self.linear(self.linear(angles.cos().cat(angles.sin(),dim=-1),'time_embed.0').silu(),'time_embed.2')
        for i in range(c['depth']):
            b=f'blocks.{i}';a=b+'.attention'
            sa,sca,ga,sm,scm,gm=self.linear(time.silu(),b+'.condition.1').chunk(6,dim=-1)
            z=self.norm(tokens,b+'.norm_attention')*(1+sca.unsqueeze(1))+sa.unsqueeze(1)
            qkv=self.linear(z,a+'.qkv').reshape(1,g*g,3,h,d//h).permute(2,0,3,1,4)
            q,k,v=qkv[0],qkv[1],qkv[2]
            def rope(z):
                pairs=z.reshape(1,h,g*g,d//h//2,2)
                rotated=(-pairs[...,1]).stack(pairs[...,0],dim=-1).reshape(1,h,g*g,d//h)
                return z*w[a+'.cosines']+rotated*w[a+'.sines']
            q=rope(self.norm(q,a+'.q_norm'));k=rope(self.norm(k,a+'.k_norm'))
            z=((q@k.transpose(-1,-2))*(d//h)**-.5).softmax(axis=-1)@v
            z=self.linear(z.permute(0,2,1,3).reshape(1,g*g,d),a+'.output')
            tokens=tokens+ga.unsqueeze(1)*z
            z=self.norm(tokens,b+'.norm_mlp')*(1+scm.unsqueeze(1))+sm.unsqueeze(1)
            gate,value=self.linear(z,b+'.mlp.input').chunk(2,dim=-1)
            tokens=tokens+gm.unsqueeze(1)*self.linear(gate.silu()*value,b+'.mlp.output')
        shift,scale=self.linear(time.silu(),'final_condition.1').chunk(2,dim=-1)
        z=self.norm(tokens,'final_norm')*(1+scale.unsqueeze(1))+shift.unsqueeze(1)
        z=self.linear(z,'final_projection').reshape(1,g,g,p,p,3)
        return z.permute(0,5,1,3,2,4).reshape(1,3,128,128).contiguous()
