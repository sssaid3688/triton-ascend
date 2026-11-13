"""Explicit NPU correctness checks for single-axis lifting and bounded outer axes."""
from __future__ import annotations
import json, os, traceback
from pathlib import Path
if __name__ != "__main__":
    import unittest
    raise unittest.SkipTest("Run explicitly on an available NPU.")
ROOT=Path("/home/oyhj/triton_cann")
os.environ["TRITON_CACHE_DIR"]=str(ROOT/".triton/lift-device-cache")
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def broadcast2(P,O,N,SHIFT:tl.constexpr,D:tl.constexpr,B:tl.constexpr,
               R:tl.constexpr,TRANS:tl.constexpr,FULL:tl.constexpr,
               PERIODIC:tl.constexpr,INDEPENDENT:tl.constexpr):
    c=tl.arange(0,B)[None,:]
    row=tl.arange(0,R)[:,None]
    logical=tl.broadcast_to(c,(R,B))
    x=SHIFT+logical
    address=x%D
    if INDEPENDENT:
        address=(x//D)*(D+7)+x%D+((row+1)//2)*97
    address=tl.broadcast_to(address,(R,B))
    active=tl.broadcast_to(logical<N,(R,B))
    if PERIODIC:
        active=active & ((x%D)<max(1,D//2))
    other=tl.full((R,B),-17.,tl.float32)
    if FULL: other=-17.+row.to(tl.float32)*0.5+logical.to(tl.float32)*0.25
    outidx=row*B+logical
    if TRANS:
        address=tl.trans(address)
        active=tl.trans(active)
        other=tl.trans(other)
        outidx=tl.trans(logical*R+row)
    v=tl.load(P+address,active,other=other)
    tl.store(O+outidx,v)

@triton.jit
def broadcast3(P,O,N,SHIFT:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    c=tl.arange(0,B)[None,:,None]
    row=tl.arange(0,2)[:,None,None]
    depth=tl.arange(0,2)[None,None,:]
    x=SHIFT+c
    address=tl.broadcast_to(x//D,(2,B,2))
    mask=tl.broadcast_to((c<N)&(x%D<max(1,D//2)),(2,B,2))
    other=-17.+row.to(tl.float32)*0.5+c.to(tl.float32)*0.25+depth.to(tl.float32)*0.125
    v=tl.load(P+address,mask,other=other)
    tl.store(O+row*B*2+c*2+depth,v)

cases=[
 dict(name="rank2_full_other",b=8,d=3,n=7,s=0,trans=False,full=True,periodic=True,independent=False),
 dict(name="rank2_phase_tail",b=128,d=24,n=113,s=1,trans=False,full=True,periodic=True,independent=False),
 dict(name="transpose_full_other",b=8,d=3,n=7,s=1,trans=True,full=True,periodic=True,independent=False),
 dict(name="rank3_full_other",b=8,d=3,n=7,s=1,rank3=True),
 dict(name="two_independent_axes",b=8,d=3,n=7,s=1,trans=False,full=True,periodic=True,independent=True),
 dict(name="aligned_rank2",b=128,d=16,n=128,s=0,trans=False,full=False,periodic=False,independent=False),
]
records=[]
cpu=(torch.arange(10000,dtype=torch.int64)*37%10007).to(torch.float32)/16
inp=cpu.to("npu:0")
for case in cases:
    rec=dict(case)
    try:
        b,d,n,s=case["b"],case["d"],case["n"],case["s"]
        outshape=(2,b,2) if case.get("rank3") else ((b,2) if case["trans"] else (2,b))
        count=1
        for x in outshape:count*=x
        out=torch.full((count+16,),-12345.,dtype=torch.float32,device="npu:0")
        c=torch.arange(b,dtype=torch.int64)
        x=s+c;mask=(c<n)&(x%d<max(1,d//2))
        if case.get("rank3"):
            broadcast3[(1,)](inp,out,n,SHIFT=s,D=d,B=b)
            other=-17.+torch.arange(2)[:,None,None]*0.5+c[None,:,None]*0.25+torch.arange(2)[None,None,:]*0.125
            wanted=torch.where(mask[None,:,None],cpu[x//d][None,:,None],other).expand(2,b,2)
        else:
            broadcast2[(1,)](inp,out,n,SHIFT=s,D=d,B=b,R=2,TRANS=case["trans"],FULL=case["full"],PERIODIC=case["periodic"],INDEPENDENT=case["independent"])
            address=(x//d)*(d+7)+x%d if case["independent"] else x%d
            address=address[None,:].expand(2,b)
            if case["independent"]:address=address+((torch.arange(2)[:,None]+1)//2)*97
            mm=mask if case["periodic"] else c<n
            other=-17.+torch.arange(2)[:,None]*0.5+c[None,:]*0.25 if case["full"] else torch.full((2,b),-17.)
            wanted=torch.where(mm[None,:],cpu[address],other)
            if case["trans"]:wanted=wanted.t().contiguous()
        torch.npu.synchronize()
        got=out.cpu()
        if not torch.equal(got[:count].reshape(outshape),wanted):
            bad=torch.nonzero(got[:count]!=wanted.reshape(-1)).flatten()[:16]
            raise AssertionError("value mismatch at "+str(bad.tolist())+" got="+str(got[bad].tolist())+" want="+str(wanted.reshape(-1)[bad].tolist()))
        if not torch.all(got[count:]==-12345.):raise AssertionError("output guard overwritten")
        rec["status"]="PASS"
    except Exception as exc:
        rec["status"]="FAIL";rec["error"]=str(exc);rec["traceback"]=traceback.format_exc()[-6000:]
    records.append(rec)
    print(case["name"],rec["status"],rec.get("error",""),flush=True)
report=ROOT/"work-tests/device-broadcast-lift.json"
report.write_text(json.dumps(records,indent=2))
print("REPORT",report,"PASS",sum(x["status"]=="PASS" for x in records),"/",len(records),flush=True)
