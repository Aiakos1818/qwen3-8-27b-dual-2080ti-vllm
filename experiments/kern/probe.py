import os, time, torch, flashinfer, inspect
print("  flashinfer", flashinfer.__version__)
from flashinfer import BatchPrefillWithPagedKVCacheWrapper as W
print("  构造签名:", [p.name for p in inspect.signature(W.__init__).parameters.values()])
dev="cuda:0"; torch.cuda.set_device(dev)
NKV, NQ, HD = 2, 12, 256
KV_LEN, Q_LEN, PS = 250000, 6, 16
npages=(KV_LEN+PS-1)//PS; maxp=npages+8
torch.manual_seed(0)
q=(torch.randn(Q_LEN,NQ,HD,dtype=torch.float16,device=dev)*0.1)
kv=torch.randn(maxp,2,PS,NKV,HD,dtype=torch.float16,device=dev)*0.05
kv8=kv.to(torch.float8_e4m3fn)
qi=torch.tensor([0,Q_LEN],dtype=torch.int32,device=dev)
pi=torch.tensor([0,npages],dtype=torch.int32,device=dev)
pidx=torch.arange(npages,dtype=torch.int32,device=dev)
last=torch.tensor([KV_LEN-(npages-1)*PS],dtype=torch.int32,device=dev)
for mb in (64, 512, 2048):
  ws=torch.empty(mb*1024*1024,dtype=torch.uint8,device=dev)
  w=W(ws, kv_layout="NHD")
  try:
    w.plan(qi,pi,pidx,last,NQ,NKV,HD,PS,head_dim_vo=HD,causal=True,
           q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn)
  except Exception as e:
    print(f"  ws={mb}MB plan 失败: {type(e).__name__}: {str(e)[:200]}"); continue
  pi_info = getattr(w,"_plan_info",None)
  print(f"  ws={mb}MB plan_info={pi_info}")
  for _ in range(3):
    o=w.run(q,kv8); torch.cuda.synchronize()
  ts=[]
  for _ in range(10):
    torch.cuda.synchronize(); t=time.perf_counter(); o=w.run(q,kv8); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1e3)
  ts.sort(); med=ts[len(ts)//2]
  print(f"  ws={mb}MB  run 中位 {med:.3f} ms  min {ts[0]:.3f}  → {256/med*1000/1e3:.0f} GB/s")
