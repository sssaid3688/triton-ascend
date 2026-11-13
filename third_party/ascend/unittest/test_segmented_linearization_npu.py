"""End-to-end NPU correctness and timing for segmented linearization.

Use --baseline to load the preserved Python/native compiler and isolated cache.
The handwritten comparison uses explicit head / complete-period rows / tail.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

if __name__ != "__main__":
    import unittest
    raise unittest.SkipTest("Explicit NPU CLI; run with --report")

sys.path.insert(0,str(Path(__file__).resolve().parent))
ROOT=Path("/home/oyhj/triton_cann")
parser=argparse.ArgumentParser()
parser.add_argument("--baseline",action="store_true")
parser.add_argument("--candidate",action="store_true")
parser.add_argument("--edges",action="store_true")
parser.add_argument("--bounded",action="store_true")
parser.add_argument("--diagnostic-small",action="store_true")
parser.add_argument("--aligned-only",action="store_true")
parser.add_argument("--cache-suffix",default="-1598f6da")
parser.add_argument("--smoke",action="store_true")
parser.add_argument("--quick",action="store_true")
parser.add_argument("--ordinary-only",action="store_true")
parser.add_argument("--benchmark",action="store_true")
parser.add_argument("--perf-targets",action="store_true")
parser.add_argument("--perf-target-case",choices=["all","packed65","packed65-periodic"],default="all")
parser.add_argument("--benchmark-mode",choices=["packed","rem","div","all"],default="packed")
parser.add_argument("--report",type=Path,required=True)
args=parser.parse_args()
if args.baseline:
    sys.path.insert(0,str(ROOT/"baseline/python"))
    os.environ["TRITON_CACHE_DIR"]=str(ROOT/"baseline/cache/triton")
    os.environ["TRITON_HOME"]=str(ROOT/"baseline")
    os.environ["ASCEND_CACHE_PATH"]=str(ROOT/"baseline/cache/ascend")
    os.environ["TRITON_DUMP_DIR"]=str(ROOT/"baseline/logs/dump")
elif args.candidate:
    sys.path.insert(0,str(ROOT/"work-tests/candidate-python"))
    os.environ["TRITON_CACHE_DIR"]=str(ROOT/("work-tests/candidate-cache"+args.cache_suffix)/"triton")
    os.environ["TRITON_HOME"]=str(ROOT/("work-tests/candidate-cache"+args.cache_suffix))
else:
    os.environ["TRITON_CACHE_DIR"]=str(ROOT/".triton/device-test-cache")

import torch
import torch_npu
if args.benchmark and not args.aligned_only:
    torch.set_num_threads(1)
import triton
import triton.language as tl
import triton._C.libtriton as _native
import hashlib
NATIVE_SHA=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()

@triton.jit
def ordinary(P,O,N,shift, D:tl.constexpr,K:tl.constexpr,SHIFT:tl.constexpr,
             STATIC:tl.constexpr,MODE:tl.constexpr,PERIODIC:tl.constexpr,
             TENSOR_OTHER:tl.constexpr,B:tl.constexpr,NAN_OTHER:tl.constexpr=False,I64:tl.constexpr=False,UNMASKED:tl.constexpr=False):
    i=tl.program_id(0)*B+tl.arange(0,B)
    if I64: i=i.to(tl.int64)
    s=SHIFT if STATIC else shift
    x=s+K*i
    q=x//D
    r=x%D
    if MODE==0: address=r
    elif MODE==1: address=q
    else: address=q*(D+7)+r
    active=x<s+K*N
    if PERIODIC: active=active & (r<max(1,D//2))
    other=-17.0
    if NAN_OTHER: other=float("nan")
    if TENSOR_OTHER: other=-17.0+i.to(tl.float32)*0.25
    if UNMASKED:
        v=tl.load(P+address)
        tl.store(O+i,v)
    else:
        v=tl.load(P+address,active,other=other)
        tl.store(O+i,v,i<N)

@triton.jit
def handwritten(P,O,N,shift,D:tl.constexpr,SHIFT:tl.constexpr,STATIC:tl.constexpr,
                MODE:tl.constexpr,PERIODIC:tl.constexpr,TENSOR_OTHER:tl.constexpr,
                B:tl.constexpr,PC:tl.constexpr,PQ:tl.constexpr):
    start=tl.program_id(0)*B
    s=SHIFT if STATIC else shift
    xs=start+s
    phase=xs%D
    left=tl.minimum(tl.maximum(N-start,0),B)
    h=tl.minimum(left,D-phase)
    h=tl.where(phase==0,0,h)
    c=tl.arange(0,PC)
    hlogical=start+c
    hq=xs//D
    hc=phase+c
    if MODE==0: ha=hc
    elif MODE==1: ha=tl.full((PC,),hq,tl.int32)
    else: ha=hq*(D+7)+hc
    hm=c<h
    hload=hm
    if PERIODIC: hload=hload & (hc<max(1,D//2))
    ho=-17.0
    if TENSOR_OTHER: ho=-17.0+hlogical.to(tl.float32)*0.25
    hv=tl.load(P+ha,hload,other=ho)
    tl.store(O+hlogical,hv,hm)
    qvalid=(left-h)//D
    bq=(xs+h)//D
    rr=tl.arange(0,PQ)[:,None]
    cc=tl.arange(0,PC)[None,:]
    logical=start+h+rr*D+cc
    if MODE==0: address=tl.broadcast_to(cc,(PQ,PC))
    elif MODE==1: address=tl.broadcast_to(bq+rr,(PQ,PC))
    else: address=(bq+rr)*(D+7)+cc
    active=(rr<qvalid)&(cc<D)
    loadmask=active
    if PERIODIC: loadmask=loadmask & (cc<max(1,D//2))
    other=-17.0
    if TENSOR_OTHER: other=-17.0+logical.to(tl.float32)*0.25
    v=tl.load(P+address,loadmask,other=other)
    tl.store(O+logical,v,active)
    t=left-h-qvalid*D
    ti=start+h+qvalid*D+c
    tq=bq+qvalid
    if MODE==0: ta=c
    elif MODE==1: ta=tl.full((PC,),tq,tl.int32)
    else: ta=tq*(D+7)+c
    tm=c<t
    tload=tm
    if PERIODIC: tload=tload & (c<max(1,D//2))
    tother=-17.0
    if TENSOR_OTHER: tother=-17.0+ti.to(tl.float32)*0.25
    tv=tl.load(P+ta,tload,other=tother)
    tl.store(O+ti,tv,tm)

@triton.jit
def bounded_intervals(P,O,N, D:tl.constexpr,SHIFT:tl.constexpr,TENSOR_OTHER:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=SHIFT+i
    q=x//D
    r=x%D
    address=q*(D+7)+r
    active=(x>=SHIFT+5)&(x<N)&(r>=1)&(r<max(1,D//2))
    other=float("nan")
    if TENSOR_OTHER: other=-17.0+i.to(tl.float32)*0.25
    v=tl.load(P+address,active,other=other)
    tl.store(O+i,v,i<N)

def timer(fn,repeats=20):
    import triton.testing
    return [1000*x for x in triton.testing.do_bench(fn,warmup=20,rep=80,quantiles=[0.5,0.9,0.99])]

def check_one(d,b,n,k,s,static,mode,periodic,tensor_other,manual=False,nan_other=False,i64=False):
    padded=max(b,triton.cdiv(n,b)*b)
    maxx=s+k*(padded+128)
    maxa=maxx//d*(d+7)+d+128
    cpu_in=(torch.arange(maxa,dtype=torch.int64)*37%10007).to(torch.float32)/16
    inp=cpu_in.to("npu:0")
    out=torch.full((padded+32,),-12345.0,dtype=torch.float32,device="npu:0")
    i=torch.arange(n,dtype=torch.int64)
    x=s+k*i
    r,q=x%d,x//d
    address=r if mode==0 else q if mode==1 else q*(d+7)+r
    wanted=cpu_in[address]
    if args.bounded:
        other=-17.0+i.to(torch.float32)*0.25 if tensor_other else torch.full((n,),float("nan"))
        active=(x>=s+5)&(x<n)&(r>=1)&(r<max(1,d//2))
        wanted=torch.where(active,wanted,other)
    elif periodic:
        other=-17.0+i.to(torch.float32)*0.25 if tensor_other else torch.full((n,),float("nan") if nan_other else -17.0)
        wanted=torch.where(r<max(1,d//2),wanted,other)
    meta=dict(D=d,SHIFT=s,STATIC=static,MODE=mode,PERIODIC=periodic,TENSOR_OTHER=tensor_other,B=b)
    if manual:
        launch=lambda: handwritten[(max(1,triton.cdiv(n,b)),)](inp,out,n,s,PC=triton.next_power_of_2(d),PQ=triton.next_power_of_2(max(1,b//d)),optimize_dynamic_offset=True,**meta)
    else:
        launch=lambda: ordinary[(max(1,triton.cdiv(n,b)),)](inp,out,n,s,K=k,NAN_OTHER=nan_other,I64=i64,UNMASKED=args.aligned_only,optimize_dynamic_offset=True,**meta)
    if args.bounded:
        launch=lambda: bounded_intervals[(max(1,triton.cdiv(n,b)),)](inp,out,n,D=d,SHIFT=s,TENSOR_OTHER=tensor_other,B=b,optimize_dynamic_offset=True)
    kernel=launch()
    torch.npu.synchronize()
    irbase=ROOT/("baseline" if args.baseline else "work-tests")/"device-ir"
    label=f"d{d}_b{b}_n{n}_k{k}_s{s}_static{static}_mode{mode}_periodic{periodic}_tensor{tensor_other}_nan{nan_other}_i64{i64}_manual{manual}_unmasked{args.aligned_only}_optTrue"
    irdir=irbase/label
    irdir.mkdir(parents=True,exist_ok=True)
    for stage in ("ttir","ttadapter","bcmlir"):
        if stage in kernel.asm and isinstance(kernel.asm[stage],str):
            (irdir/(stage+".mlir")).write_text(kernel.asm[stage])
    converted=None
    if not manual and "ttir" in kernel.asm:
        import subprocess
        from test_segmented_linearization import pointer_is_structured
        opt=ROOT/("baseline/binaries/triton-opt" if args.baseline else "work-tests/final-triton-opt")
        converted_ir=subprocess.run([str(opt),f"--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true"],input=kernel.asm["ttir"],text=True,capture_output=True)
        (irdir/"structured.mlir").write_text(converted_ir.stdout)
        (irdir/"structured.stderr").write_text(converted_ir.stderr)
        converted=converted_ir.returncode==0 and pointer_is_structured(converted_ir.stdout)
    got=out.cpu()
    if not torch.all((got[:n]==wanted)|(torch.isnan(got[:n])&torch.isnan(wanted))):
        indices=torch.nonzero(got[:n]!=wanted).flatten()[:12].tolist()
        raise AssertionError(f"wrong output at {indices}: got={got[indices].tolist()} want={wanted[indices].tolist()} IR={irdir}")
    if not torch.all(got[n:]==-12345.0):
        raise AssertionError("output tail guard was overwritten")
    timing=timer(launch) if args.benchmark else None
    return {"microseconds":None if timing is None else timing[0],"p90_microseconds":None if timing is None else timing[1],"p99_microseconds":None if timing is None else timing[2],"structured":converted,"actual_indirect_load": "triton_indirect_load" in kernel.asm.get("ttadapter",""),"actual_mix_simd_simt": "mix_simd_simt" in kernel.asm.get("ttadapter",""),"optimize_dynamic_offset":True,"ir_dir":str(irdir)}

rows=[]
cases=[(16,128,371,1,0,True,2,False,False)]
if not args.smoke:
    cases=[]
    for d in [3,5,7,16,24,63,65,127,129]:
        for s,static in [(0,True),(1,True),(1,False)]:
            for mode in [0,1,2]:
                for periodic in [False,True]:
                    cases.append((d,128,371,1,s,static,mode,periodic,False))
    cases += [(d,128,371,k,1,True,2,True,False) for d in [16,24] for k in [2,3,8]]
    cases += [(d,128,371,1,1,False,2,True,True) for d in [3,16,65]]
if args.quick:
    cases=[(d,128,371,1,1,static,2,periodic,False) for d in [3,16,65] for static in [True,False] for periodic in [False,True]]
    cases += [(d,128,371,1,1,False,2,True,True) for d in [3,16,65]]
if args.benchmark:
    modes=[0,1,2] if args.benchmark_mode=="all" else [dict(rem=0,div=1,packed=2)[args.benchmark_mode]]
    cases=[(d,1024,262131,1,s,static,mode,False,False) for mode in modes for d in [16,24,65] for s,static in [(0,True),(1,False)]]
if args.edges:
    cases=[(d,128,n,1,1,True,2,True,tensor) for d in [3,16,65,256] for n in [0,1,127,128] for tensor in [False,True]]
if args.aligned_only:
    cases=[(16,1024,262144,1,0,True,mode,False,False) for mode in [0,1,2]]
if args.bounded:
    cases=[(d,128,127,1,1,True,2,True,tensor) for d in [7,65] for tensor in [False,True]]
if args.diagnostic_small:
    cases=[(3,b,n,1,s,True,mode,True,True) for b,n,s,mode in [(8,7,0,0),(8,7,1,0),(128,127,0,0),(128,127,1,0),(8,7,0,1),(128,127,1,1)]]
if args.perf_targets:
    cases=[(16,1024,262131,1,0,True,0,False,False),(24,1024,262131,1,0,True,0,False,False),(65,1024,262131,1,0,True,2,False,False)]
if args.perf_targets and args.perf_target_case=="packed65":
    cases=[(65,1024,262131,1,0,True,2,False,False)]
if args.perf_targets and args.perf_target_case=="packed65-periodic":
    cases=[(65,1024,262131,1,0,True,2,True,False)]
for case in cases:
    d,b,n,k,s,static,mode,periodic,tensor_other=case
    for manual in ([False,True] if k==1 and not args.ordinary_only and not args.edges and not args.bounded and not args.diagnostic_small else [False]):
        row=dict(D=d,B=b,N=n,K=k,S=s,static=static,mode=mode,periodic=periodic,tensor_other=tensor_other,manual=manual)
        try:
            row.update(check_one(*case,manual=manual,nan_other=args.edges and not tensor_other,i64=args.edges and d in [65,256]))
            row["ok"]=True
        except Exception as exc:
            row["ok"]=False
            row["error"]=f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(json.dumps(row),flush=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({"baseline":args.baseline,"python":sys.executable,"triton":triton.__file__,"torch":torch.__version__,"torch_npu":torch_npu.__version__,"native_path":_native.__file__,"native_sha256":NATIVE_SHA,"cpu_threads":torch.get_num_threads(),"optimize_dynamic_offset":True,"rows":rows},indent=2))
print("SUMMARY",len(rows),"passed",sum(x["ok"] for x in rows),"failed",sum(not x["ok"] for x in rows),flush=True)
sys.exit(0 if all(x["ok"] for x in rows) else 1)
