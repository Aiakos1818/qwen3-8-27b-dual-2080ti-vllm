import torch, flashinfer, time
dev="cuda:0"; torch.cuda.set_device(dev)
H_Q,H_KV,D,PAGE=12,2,256,16
KV=250000; npages=(KV+PAGE-1)//PAGE+8
kv=torch.randint(0,255,(npages,2,PAGE,H_KV,D),dtype=torch.uint8,device=dev).view(torch.float8_e4m3fn)
qi=torch.tensor([0,1],dtype=torch.int32,device=dev)
pidx=torch.arange(npages,dtype=torch.int32,device=dev)
last=torch.tensor([KV-(npages-1)*PAGE],dtype=torch.int32,device=dev)
ws=torch.empty(768*1024*1024,dtype=torch.uint8,device=dev)
q=torch.randn(1,H_Q,D,dtype=torch.float16,device=dev)*0.5
for tc in (False,True):
    try:
        t0=time.time()
        w=flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD", use_tensor_cores=tc)
        w.plan(indptr=qi, indices=pidx, last_page_len=last, num_qo_heads=H_Q, num_kv_heads=H_KV,
               head_dim=D, page_size=PAGE, q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn)
        o=w.run(q,kv); torch.cuda.synchronize()
        print(f"  tensor_cores={tc}: 首次(含JIT) {time.time()-t0:.0f} s, plan={w._plan_info[:5]}", flush=True)
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(20): w.run(q,kv)
        b.record(); torch.cuda.synchronize()
        print(f"  ✓ tensor_cores={tc}: {a.elapsed_time(b)/20:6.3f} ms/call  (对比 prefill 路径 1.45 ms)", flush=True)
    except Exception as e:
        print(f"  ✗ tensor_cores={tc}: {type(e).__name__}: {str(e)[:140]}", flush=True)
print("  DONE", flush=True)
