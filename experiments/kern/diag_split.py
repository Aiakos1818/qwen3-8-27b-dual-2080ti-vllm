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
H_Q,H_KV,D,QL,PAGE,MR=12,2,256,6,16,48
torch.manual_seed(0)
KV=512; nchunks=2; per=256
npages=(KV+PAGE-1)//PAGE
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
# reference S (scaled, masked) and O
kf=kv[:,0].reshape(-1,H_KV,D)[:KV].float(); vf=kv[:,1].reshape(-1,H_KV,D)[:KV].float()
rows=torch.arange(QL,device=dev)[:,None]; cols=torch.arange(KV,device=dev)[None,:]
lim=KV-QL+rows+1
Sref=torch.full((QL,H_Q,KV),float("-inf"),device=dev)
for h in range(H_Q):
    kh=h//(H_Q//H_KV)
    s=(q[:,h].float()@kf[:,kh].T)*(1.0/16.0)
    Sref[0,h]=s.masked_fill(cols.squeeze(0)>=lim.squeeze(1),float("-inf")).transpose(0,1) if False else s
# per-(row,head) error of the merged output
ref=torch.zeros(QL,H_Q,D,device=dev)
for h in range(H_Q):
    kh=h//(H_Q//H_KV)
    p=torch.softmax(Sref[:,h].masked_fill(cols>=lim,float("-inf")),dim=-1)
    ref[:,h]=p@vf[:,kh]
err=(out.float()-ref).abs()
print(f"  最终输出 max|err| {err.max().item():.5f}  rel {err.max().item()/ref.abs().max().item():.2e}")
per_qr=err.amax(dim=(1,2)); per_h=err.amax(dim=(0,2))
print(f"  按 query 行: {[f'{x:.4f}' for x in per_qr.tolist()]}")
print(f"  按 head    : {[f'{x:.4f}' for x in per_h.tolist()]}")
# kernel B 的 l vs 参考 sum exp
l_ref=[]
for h in range(H_Q):
    kh=h//(H_Q//H_KV)
    l_ref.append(torch.exp(Sref[:,h].masked_fill(cols>=lim,float("-inf"))).sum(-1))
# 打印 kernel B 写的 l（按 chunk 合并后 = merge 的 lg）
lg=torch.zeros(48*2,device=dev)
for kvh in range(2):
    for r in range(48):
        mg=max(pl[((c*H_KV+kvh)*MR+r)*2].item() for c in range(nchunks))
        lg[kvh*48+r]=sum(pl[((c*H_KV+kvh)*MR+r)*2+1].item()*torch.exp(torch.tensor(pl[((c*H_KV+kvh)*MR+r)*2].item()-mg)).item() for c in range(nchunks))
for kvh in range(2):
    for qr in range(2):
        h=kvh*6+qr*0+0
        r=qr*6+0
        lref_sum=sum(l_ref[hh][qr].item() for hh in range(kvh*6,(kvh+1)*6))
        print(f"   kvh={kvh} qr={qr}: kernelB lg(6 head 之和)={sum(lg[kvh*48+qr*6+g].item() for g in range(6)):.3f}  参考={lref_sum:.3f}")
