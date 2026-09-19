import time, torch, flashinfer
import os
dev="cuda:0"; torch.cuda.set_device(dev)
H_Q,H_KV,D,QL=12,2,256,6
KV_LEN,PAGE=250000,16
npages=(KV_LEN+PAGE-1)//PAGE
q=(torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)*0.5)
kv=torch.randn(npages,2,PAGE,H_KV,D,dtype=torch.float16,device=dev).to(torch.float8_e4m3fn)
qi=torch.tensor([0,QL],dtype=torch.int32,device=dev)
pi=torch.tensor([0,npages],dtype=torch.int32,device=dev)
pidx=torch.arange(npages,dtype=torch.int32,device=dev)
last=torch.tensor([KV_LEN-(npages-1)*PAGE],dtype=torch.int32,device=dev)
ws=torch.empty(512*1024*1024,dtype=torch.uint8,device=dev)
w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,kv_layout="NHD")
t0=time.time()
w.plan(qi,pi,pidx,last,H_Q,H_KV,D,PAGE,head_dim_vo=D,causal=True,
       q_data_type=torch.float16,kv_data_type=torch.float8_e4m3fn)
print(f"  JIT 构建+plan 耗时 {time.time()-t0:.1f}s  plan_info={w._plan_info[:5]}")
o=w.run(q,kv); torch.cuda.synchronize()
for _ in range(3): w.run(q,kv)
torch.cuda.synchronize()
st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
for _ in range(20): w.run(q,kv)
en.record(); torch.cuda.synchronize()
ms=st.elapsed_time(en)/20
print(f"  ★ 中位 {ms:.3f} ms  {npages*2*PAGE*H_KV*D/ms/1e6:.0f} GB/s")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    w.run(q,kv); torch.cuda.synchronize()
for e in prof.key_averages():
    if "Prefill" in e.key or "Merge" in e.key:
        print(f"  [{e.key[:70]}] n={e.count}")
for ev in prof.events():
    if "BatchPrefillWithPagedKVCacheKernel" in ev.name:
        nm=ev.name
        print(f"  block={nm}")
        break
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe2_trace.txt"),"w") as f:
    f.write(prof.key_averages().table(sort_by="cuda_time_total",row_limit=8))
print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe2_trace.txt")).read()[:1200])
