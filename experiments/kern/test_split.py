import torch, sys
import os
from torch.utils.cpp_extension import load_inline
cuda = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mq_split.cu")).read()
mod = load_inline(name="mqsplit", cpp_sources="""
#include <c10/cuda/CUDAStream.h>
void mq_scores_launch(const void*,const void*,const void*,void*,void*,int,int,int,void*);
void mq_pv_launch(const void*,const void*,const void*,const void*,void*,void*,int,int,int,void*);
void mq_merge_launch(const void*,const void*,void*,int,void*);
int mq_smem_a(); int mq_smem_b();
void run_all(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor sbuf,
             torch::Tensor mchunk, torch::Tensor po, torch::Tensor pl, torch::Tensor out,
             int64_t kv_len, int64_t nchunks, int64_t per) {
  void* st = (void*)c10::cuda::getCurrentCUDAStream();
  mq_scores_launch(q.data_ptr(), kv.data_ptr(), idx.data_ptr(), sbuf.data_ptr(),
                   mchunk.data_ptr(), (int)kv_len, (int)nchunks, (int)per, st);
  mq_pv_launch(kv.data_ptr(), idx.data_ptr(), sbuf.data_ptr(), mchunk.data_ptr(),
               po.data_ptr(), pl.data_ptr(), (int)kv_len, (int)nchunks, (int)per, st);
  mq_merge_launch(po.data_ptr(), pl.data_ptr(), out.data_ptr(), (int)nchunks, st);
}
""", cuda_sources=cuda, functions=["run_all","mq_smem_a","mq_smem_b"],
   extra_cuda_cflags=["-O3","-arch=sm_75","--use_fast_math"], verbose=False)

dev="cuda:0"; torch.cuda.set_device(dev)
H_Q,H_KV,D,QL,PAGE=12,2,256,6,16
MR=48
print(f"  smem A={mod.mq_smem_a()/1024:.1f} KB  B={mod.mq_smem_b()/1024:.1f} KB")

def ref(q, kv, kv_len):
    k=kv[:,0].reshape(-1,H_KV,D)[:kv_len].float(); v=kv[:,1].reshape(-1,H_KV,D)[:kv_len].float()
    out=torch.zeros(QL,H_Q,D,dtype=torch.float32,device=dev)
    rows=torch.arange(QL,device=dev)[:,None]; cols=torch.arange(kv_len,device=dev)[None,:]
    lim=kv_len-QL+rows+1
    for h in range(H_Q):
        kh=h//(H_Q//H_KV)
        s=(q[:,h].float()@k[:,kh].T)*(1.0/16.0)
        p=torch.softmax(s.masked_fill(cols>=lim,float("-inf")),dim=-1)
        out[:,h]=p@v[:,kh]
    return out

def run(q,kv,idx,kv_len,nchunks,per):
    sbuf=torch.zeros(nchunks*H_KV*MR*per,dtype=torch.float16,device=dev)
    mchunk=torch.zeros(nchunks*H_KV*MR,dtype=torch.float32,device=dev)
    po=torch.zeros(nchunks*H_KV*MR*D,dtype=torch.float32,device=dev)
    pl=torch.zeros(nchunks*H_KV*MR*2,dtype=torch.float32,device=dev)
    out=torch.zeros(QL,H_Q,D,dtype=torch.float16,device=dev)
    mod.run_all(q,kv,idx,sbuf,mchunk,po,pl,out,kv_len,nchunks,per)
    return out

torch.manual_seed(0)
print("  == 正确性 kv_len=512 ==")
KV=512
kv=torch.randn(KV//PAGE,2,PAGE,H_KV,D,dtype=torch.float16,device=dev)*0.1
kv=kv.to(torch.float8_e4m3fn)
idx=torch.arange(kv.shape[0],dtype=torch.int32,device=dev)
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5
got=run(q,kv,idx,KV,2,256).float(); r=ref(q,kv,KV)
err=(got-r).abs().max().item(); rel=err/r.abs().max().item()
print(f"   max|err| {err:.5f}  rel {rel:.2e}  {'OK' if rel<3e-2 else 'FAIL'}")

print("  == 速度 kv_len=250000 ==")
KV=250000; npages=(KV+PAGE-1)//PAGE
per=64*((npages*PAGE//64+67)//68); nchunks=(npages*PAGE+per-1)//per
kv=torch.randint(0,200,(npages,2,PAGE,H_KV,D),dtype=torch.uint8,device=dev)
idx=torch.arange(npages,dtype=torch.int32,device=dev)
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5
sbuf=torch.zeros(nchunks*H_KV*MR*per,dtype=torch.float16,device=dev)
mchunk=torch.zeros(nchunks*H_KV*MR,dtype=torch.float32,device=dev)
po=torch.zeros(nchunks*H_KV*MR*D,dtype=torch.float32,device=dev)
pl=torch.zeros(nchunks*H_KV*MR*2,dtype=torch.float32,device=dev)
out=torch.zeros(QL,H_Q,D,dtype=torch.float16,device=dev)
def call(): mod.run_all(q,kv,idx,sbuf,mchunk,po,pl,out,KV,nchunks,per)
call(); torch.cuda.synchronize()
st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
for _ in range(20): call()
en.record(); torch.cuda.synchronize()
ms=st.elapsed_time(en)/20
nb=npages*2*PAGE*H_KV*D
print(f"   per={per} nchunks={nchunks}  {ms:.3f} ms   KV读 {nb/1e6:.0f} MB + S 写读 {2*nchunks*H_KV*MR*per*2/1e6:.0f} MB → 有效 {nb/ms/1e6:.0f} GB/s(KV)")
