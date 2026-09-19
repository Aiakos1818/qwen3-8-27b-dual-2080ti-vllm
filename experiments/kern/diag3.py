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
def ref_out(q,kv,KV):
    kf=kv[:,0].reshape(-1,H_KV,D)[:KV].float(); vf=kv[:,1].reshape(-1,H_KV,D)[:KV].float()
    cols=torch.arange(KV,device=dev)[None,:]; lim=KV-QL+torch.arange(QL,device=dev)[:,None]+1
    out=torch.zeros(QL,H_Q,D,device=dev)
    for h in range(H_Q):
        kh=h//GROUP
        S=((q[:,h].float()@kf[:,kh].T)*(1.0/16.0)).masked_fill(cols>=lim,float("-inf"))
        out[:,h]=torch.softmax(S,dim=-1)@vf[:,kh]
    return out
torch.manual_seed(0)
for (KV,nchunks,per) in ((512,1,512),(512,2,256),(1024,4,256),(512,8,64)):
    npages=(KV+PAGE-1)//PAGE+4
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
    r=ref_out(q,kv,KV)
    e=(out.float()-r).abs()
    print(f"  KV={KV} nchunks={nchunks} per={per}: max|err| {e.max().item():.5f} rel {e.max().item()/r.abs().max().item():.2e}")
    if e.max().item()/r.abs().max().item() > 3e-2:
        print(f"     按行 {[f'{x:.3f}' for x in e.amax(dim=(1,2)).tolist()]}")
        print(f"     按head {[f'{x:.3f}' for x in e.amax(dim=(0,2)).tolist()]}")
