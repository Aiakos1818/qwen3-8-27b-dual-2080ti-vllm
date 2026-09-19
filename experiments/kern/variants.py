import time, torch, flashinfer
dev="cuda:0"; torch.cuda.set_device(dev)
NKV,NQ,HD=2,12,256; KV_LEN,Q_LEN=250000,6
torch.manual_seed(0)
q=(torch.randn(Q_LEN,NQ,HD,dtype=torch.float16,device=dev)*0.1)
def bench(ps, fp16red, niter=8):
  npages=(KV_LEN+ps-1)//ps; maxp=npages+4
  kv=torch.randn(maxp,2,ps,NKV,HD,dtype=torch.float16,device=dev)*0.05
  kv8=kv.to(torch.float8_e4m3fn)
  qi=torch.tensor([0,Q_LEN],dtype=torch.int32,device=dev)
  pi=torch.tensor([0,npages],dtype=torch.int32,device=dev)
  pidx=torch.arange(npages,dtype=torch.int32,device=dev)
  last=torch.tensor([KV_LEN-(npages-1)*ps],dtype=torch.int32,device=dev)
  ws=torch.empty(512*1024*1024,dtype=torch.uint8,device=dev)
  w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
  try:
    w.plan(qi,pi,pidx,last,NQ,NKV,HD,ps,head_dim_vo=HD,causal=True,use_fp16_qk_reduction=fp16red,
           q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn)
  except Exception as e:
    return f"plan 失败 {type(e).__name__}: {str(e)[:90]}"
  for _ in range(2): w.run(q,kv8)
  torch.cuda.synchronize(); ts=[]
  for _ in range(niter):
    t=time.perf_counter(); w.run(q,kv8); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1e3)
  ts.sort()
  return f"plan={w._plan_info[:5]} 中位 {ts[len(ts)//2]:.3f} ms → {256/ts[len(ts)//2]*1000/1e3:.0f} GB/s"
for ps in (16,32,64,128):
  for r in (False,True):
    print(f"  page={ps:3d} fp16red={str(r):5s}  {bench(ps,r)}")
