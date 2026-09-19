import torch
import os
from torch.utils.cpp_extension import load_inline
cuda=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mq_split.cu")).read()
mod=load_inline(name="mqsplit", cpp_sources="""
#include <c10/cuda/CUDAStream.h>
void mq_scores_launch(const void*,const void*,const void*,void*,void*,int,int,int,int,void*);
void mq_pv_launch(const void*,const void*,const void*,const void*,void*,void*,int,int,int,int,void*);
void mq_merge_launch(const void*,const void*,void*,int,void*);
void run_all(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor sbuf,
             torch::Tensor mchunk, torch::Tensor po, torch::Tensor pl, torch::Tensor out,
             int64_t kv_len,int64_t nchunks,int64_t per,int64_t npages) {
  void* st=(void*)c10::cuda::getCurrentCUDAStream();
  mq_scores_launch(q.data_ptr(),kv.data_ptr(),idx.data_ptr(),sbuf.data_ptr(),mchunk.data_ptr(),
                   (int)kv_len,(int)nchunks,(int)per,(int)npages,st);
  mq_pv_launch(kv.data_ptr(),idx.data_ptr(),sbuf.data_ptr(),mchunk.data_ptr(),po.data_ptr(),
               pl.data_ptr(),(int)kv_len,(int)nchunks,(int)per,(int)npages,st);
  mq_merge_launch(po.data_ptr(),pl.data_ptr(),out.data_ptr(),(int)nchunks,st);
}
""", cuda_sources=cuda, functions=["run_all"],
   extra_cuda_cflags=["-O3","-arch=sm_75","--use_fast_math"], verbose=False)
dev="cuda:0"; torch.cuda.set_device(dev)
H_Q,H_KV,D,QL,PAGE,MR,GROUP=12,2,256,6,16,48,6
torch.manual_seed(0)
KV=512; nchunks=2; per=256; npages=(KV+PAGE-1)//PAGE
kv=torch.randn(npages,2,PAGE,H_KV,D,dtype=torch.float16,device=dev)*0.1
kv=kv.to(torch.float8_e4m3fn)
idx=torch.arange(npages,dtype=torch.int32,device=dev)
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5
sbuf=torch.zeros(nchunks*H_KV*MR*per,dtype=torch.float16,device=dev)
mchunk=torch.zeros(nchunks*H_KV*MR,dtype=torch.float32,device=dev)
po=torch.zeros(nchunks*H_KV*MR*D,dtype=torch.float32,device=dev)
pl=torch.zeros(nchunks*H_KV*MR*2,dtype=torch.float32,device=dev)
out=torch.zeros(QL,H_Q,D,dtype=torch.float16,device=dev)
mod.run_all(q,kv,idx,sbuf,mchunk,po,pl,out,KV,nchunks,per,npages)
torch.cuda.synchronize()

kf=kv[:,0].reshape(-1,H_KV,D)[:KV].float(); vf=kv[:,1].reshape(-1,H_KV,D)[:KV].float()
cols=torch.arange(KV,device=dev)[None,:]
lim=(KV-QL+torch.arange(QL,device=dev)[:,None]+1)

# --- kernel A 的 S 对照 ---
print("  == kernel A 的 S 检查 ==")
for (qr,h) in ((0,0),(3,7),(5,11)):
    kvh=h//GROUP; hig=h%GROUP; r=qr*GROUP+hig
    s_ref=((q[qr,h].float()@kf[:,kvh].T)*(1.0/16.0)).masked_fill(cols>=lim[qr],float("-inf"))
    c0=0; s_got=sbuf[(c0*H_KV+kvh)*MR*per + r*per : (c0*H_KV+kvh)*MR*per + r*per+KV//2].float()
    d=(s_got - s_ref[:KV//2]).abs(); d=torch.nan_to_num(d,nan=0.0,posinf=0.0)
    print(f"   qr={qr} h={h}: max|S 差| {d.max().item():.5f} (前 256 key)")

# --- kernel B 的 l 对照（每 chunk 的 l 与参考）---
print("  == kernel B 的 l 检查（chunk 0, kvh=0）==")
for r in (0, 1, 7):
    qr=r//GROUP
    ls_got=pl[((0*H_KV+0)*MR+r)*2+1].item()
    ls_ref=sum(torch.exp(((q[qr,h].float()@kf[:,0].T)*(1.0/16.0)).masked_fill(cols>=lim[qr],float("-inf"))).sum().item() for h in range(0,GROUP) if False)
    r0=r
    ls_ref=torch.exp(((q[qr, r0%GROUP if False else 0].float()@kf[:,0].T)*(1.0/16.0))).sum().item() if False else None
    print(f"   r={r} l(chunk0)={ls_got:.3f}")

# --- 最终输出 ---
ref=torch.zeros(QL,H_Q,D,device=dev)
lref=[]
for h in range(H_Q):
    kh=h//GROUP
    S=(q[:,h].float()@kf[:,kh].T)*(1.0/16.0)
    S=S.masked_fill(cols>=lim,float("-inf"))
    lref.append(torch.exp(S).sum(-1))
    ref[:,h]=torch.softmax(S,dim=-1)@vf[:,kh]
err=(out.float()-ref).abs()
print(f"  == 最终输出 max|err| {err.max().item():.5f}  rel {err.max().item()/ref.abs().max().item():.2e} ==")
print(f"   按行: {[f'{x:.4f}' for x in err.amax(dim=(1,2)).tolist()]}")
print(f"   按head: {[f'{x:.4f}' for x in err.amax(dim=(0,2)).tolist()]}")
# l 的合并检查
for kvh in range(2):
    for qr in range(QL):
        r=qr*GROUP
        lg=0.0
        mg=max(pl[((c*H_KV+kvh)*MR+r)*2].item() for c in range(nchunks))
        for c in range(nchunks):
            lg += pl[((c*H_KV+kvh)*MR+r)*2+1].item()*torch.exp(torch.tensor(pl[((c*H_KV+kvh)*MR+r)*2].item()-mg)).item()
        lr=sum(lref[kvh*GROUP+g][qr].item() for g in range(GROUP))
        if qr in (0,3,5):
            print(f"   kvh={kvh} qr={qr} 合并后 l={lg:.2f}  参考(该 kvh 的 6 个 head 之和)={lr:.2f}")
