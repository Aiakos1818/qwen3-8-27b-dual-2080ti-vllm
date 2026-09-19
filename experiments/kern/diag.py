import sys, torch, triton
import os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import mqattn
dev="cuda:0"; torch.cuda.set_device(dev)
H_Q,H_KV,D,QL,PAGE=12,2,256,6,16
q=torch.randn(QL,H_Q,D,dtype=torch.float16,device=dev)
for bm in (16,32,64):
  for bn in (16,32,64):
    for dc in (1,2):
      db=D//dc
      npages=64
      kv=torch.randn(npages,2,PAGE,H_KV,D,dtype=torch.float16,device=dev)
      idx=torch.arange(npages,dtype=torch.int32,device=dev)
      pacc=torch.empty(2*H_KV*bm*D,dtype=torch.float32,device=dev)
      plse=torch.empty(2*2*H_KV*bm,dtype=torch.float32,device=dev)
      try:
        k=mqattn.mq_seg_kernel.warmup(q,kv,idx,pacc,plse,1024,
            H_Q=H_Q,H_KV=H_KV,GROUP=6,D=D,DB=db,DC=dc,QL=QL,PAGE=PAGE,
            BLOCK_M=bm,BLOCK_N=bn,NUM_SEG=2,SM_SCALE=0.06,num_warps=8,num_stages=1,grid=(2,H_KV))
        print(f"  BM={bm:3d} BN={bn:3d} dc={dc}  smem={k.metadata.shared/1024:.1f} KB")
      except Exception as e:
        print(f"  BM={bm:3d} BN={bn:3d} dc={dc}  失败: {str(e)[:120]}")
