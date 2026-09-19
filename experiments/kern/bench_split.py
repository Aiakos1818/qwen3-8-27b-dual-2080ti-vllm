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
KV=250000; npages_need=(KV+PAGE-1)//PAGE
per=64*((KV//64+67)//68); nchunks=(KV+per-1)//per
npages=npages_need+8
kv=torch.randint(0,200,(npages,2,PAGE,H_KV,D),dtype=torch.uint8,device=dev)
idx=torch.arange(npages,dtype=torch.int32,device=dev)
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5
sbuf=torch.zeros(nchunks*H_KV*MR*per,dtype=torch.float16,device=dev)
mchunk=torch.zeros(nchunks*H_KV*MR,dtype=torch.float32,device=dev)
po=torch.zeros(nchunks*H_KV*MR*D,dtype=torch.float32,device=dev)
pl=torch.zeros(nchunks*H_KV*MR*2,dtype=torch.float32,device=dev)
out=torch.zeros(QL,H_Q,D,dtype=torch.float16,device=dev)
def call(): mod.run_all(q,kv,idx,sbuf,mchunk,po,pl,out,KV,nchunks,per,npages)
call(); torch.cuda.synchronize()
st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
for _ in range(20): call()
en.record(); torch.cuda.synchronize()
ms=st.elapsed_time(en)/20
kvb=KV*2*H_KV*D   # K+V 实际读量(按 250K 计)
sb=nchunks*H_KV*MR*per*2
print(f"  per={per} nchunks={nchunks}")
print(f"  三 kernel 合计 {ms:.3f} ms   流量 = K {kvb/2/1e6:.0f} + S写 {sb/1e6:.0f} + S读 {sb/1e6:.0f} + V {kvb/2/1e6:.0f} = {(kvb+2*sb)/1e6:.0f} MB")
print(f"  有效带宽 {(kvb+2*sb)/ms/1e6:.0f} GB/s   (flashinfer 同形状 1.59 ms / 161 GB/s)")
