import torch
import os
from torch.utils.cpp_extension import load_inline
cuda=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mq_split.cu")).read()
mod=load_inline(name="mqsplit", cpp_sources="""
#include <c10/cuda/CUDAStream.h>
void mq_scores_launch(const void*,const void*,const void*,void*,void*,int,int,int,int,void*);
void run_scores(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor sbuf,
                torch::Tensor mchunk, int64_t kv_len,int64_t nchunks,int64_t per,int64_t npages) {
  mq_scores_launch(q.data_ptr(),kv.data_ptr(),idx.data_ptr(),sbuf.data_ptr(),mchunk.data_ptr(),
                   (int)kv_len,(int)nchunks,(int)per,(int)npages,(void*)c10::cuda::getCurrentCUDAStream());
}
""", cuda_sources=cuda, functions=["run_scores"],
   extra_cuda_cflags=["-O3","-arch=sm_75","--use_fast_math"], verbose=False)
dev="cuda:0"; torch.cuda.set_device(dev)
H_Q,H_KV,D,QL,PAGE,MR,GROUP=12,2,256,6,16,48,6
torch.manual_seed(0)
KV=512; nchunks=1; per=512; npages=KV//PAGE+4
kv=torch.randn(npages,2,PAGE,H_KV,D,dtype=torch.float16,device=dev)*0.1
kv8=kv.to(torch.float8_e4m3fn)
idx=torch.arange(npages,dtype=torch.int32,device=dev)
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5
sbuf=torch.full((nchunks*H_KV*MR*per,),0.0,dtype=torch.float16,device=dev)
mchunk=torch.zeros(nchunks*H_KV*MR,dtype=torch.float32,device=dev)
mod.run_scores(q,kv8,idx,sbuf,mchunk,KV,nchunks,per,npages); torch.cuda.synchronize()
kf=kv8[:,0].reshape(-1,H_KV,D)[:KV].float()
cols=torch.arange(KV,device=dev)
lim=KV-QL+torch.arange(QL,device=dev)+1
print("  == kernel A 的 S vs 参考 ==")
for (qr,h) in ((0,0),(0,5),(3,7),(5,11)):
    kvh=h//GROUP; hig=h%GROUP; r=qr*GROUP+hig
    # 参考：fp16 输入 q，fp8->fp16 的 k，与 kernel 的 e4m3_to_half 应一致
    kf16=kv8[:,0].reshape(-1,H_KV,D)[:KV].to(torch.float16)
    s_ref=((q[qr,h].to(torch.float32)@kf16[:,kvh].float().T)*(1.0/16.0))
    s_ref=torch.where(cols<lim[qr], s_ref, torch.full_like(s_ref,float("-inf")))
    s_got=sbuf[r*per:(r+1)*per].float()
    valid=cols<lim[qr]
    d=(s_got[valid]-s_ref[valid]).abs()
    bad=(s_got[~valid]!=float("-inf")).sum().item()
    print(f"   qr={qr} h={h}: 有效键 max|ΔS| {d.max().item():.6f}  均值 {d.mean().item():.6f} | 未屏蔽的无效键 {bad}")
    if d.max().item()>1e-3:
        i=d.argmax().item(); print(f"        最大差异在 key={i}: got {s_got[i].item():.5f} ref {s_ref[i].item():.5f}")
