"""IR conversion and numerical regression matrix for segmented linearization.

Run without an NPU:
  python test_segmented_linearization.py --triton-opt /path/to/triton-opt \
      --report /path/to/report.json --snapshots /path/to/ir
The evaluator executes generated memory-address IR with NumPy. Unsupported
operations are reported explicitly, never counted as successful validation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

SSA = re.compile(r"%[-a-zA-Z$._0-9]+")
MEMORY = (np.arange(262144, dtype=np.int64) * 37 % 10007).astype(np.float32) / 16
OTHER = np.float32(-17.0)

@dataclass(frozen=True)
class Case:
    block: int = 128
    divisor: int = 16
    stride: int = 1
    shift: int = 0
    operation: str = "rem"
    mask: str = "none"
    dynamic_shift: bool = False
    program_shift: bool = False
    program_id: int = 1
    other: str = "scalar"
    valid_count: int = -1
    bits: int = 32

    @property
    def effective_shift(self):
        return self.shift + (self.program_id*self.block if self.program_shift else 0)

    @property
    def count(self):
        return self.valid_count if self.valid_count>=0 else self.block-13

    @property
    def name(self):
        extra=("_pid"+str(self.program_id) if self.program_shift else "")+("_other"+self.other if self.other!="scalar" else "")+("_n"+str(self.valid_count) if self.valid_count>=0 else "")+("_i64" if self.bits==64 else "")
        return f"{self.operation}_b{self.block}_d{self.divisor}_k{self.stride}_s{self.shift}_{self.mask}_{'dyn' if self.dynamic_shift else 'static'}"+extra

    def expected(self):
        x = self.effective_shift + self.stride * np.arange(self.block, dtype=np.int64)
        r, q = x % self.divisor, x // self.divisor
        idx = r if self.operation == "rem" else q if self.operation == "div" else q * (self.divisor + 7) + r
        mask = np.ones(self.block, dtype=bool)
        if self.mask in ("linear", "combined"):
            mask &= x < self.effective_shift + self.stride*self.count
        if self.mask in ("periodic", "combined"):
            mask &= r < max(1,self.divisor//2)
        if self.mask == "quotient":
            mask &= q < max(1,self.block//self.divisor-1)
        result = (-17+np.arange(self.block,dtype=np.float32)*0.25) if self.other=="tensor" else np.full(self.block,np.nan if self.other=="nan" else OTHER,dtype=np.float32)
        result[mask] = MEMORY[idx[mask]]
        return result, idx[mask]

    def mlir(self):
        b,d,bits=self.block,self.divisor,self.bits
        typ=f"tensor<{b}xi{bits}>"
        lines=[f"module {{ tt.func public @test(%p: !tt.ptr<f32>, %length: i{bits}"+(f", %shift: i{bits}" if self.dynamic_shift else "")+f") -> tensor<{b}xf32> {{"]
        if self.program_shift:
            lines += ["%pid32 = tt.get_program_id x : i32"]
            if bits==64: lines += ["%pid = arith.extsi %pid32 : i32 to i64"]
            else: lines += ["%pid = arith.addi %pid32, %pid32 : i32"] if False else []
            pid="%pid" if bits==64 else "%pid32"
            lines += [f"%bs = arith.constant {b} : i{bits}",f"%pshift = arith.muli {pid}, %bs : i{bits}",f"%sc = arith.constant {self.shift} : i{bits}",f"%shift = arith.addi %pshift, %sc : i{bits}"]
        elif not self.dynamic_shift: lines += [f"%shift = arith.constant {self.shift} : i{bits}"]
        lines += [
            f"%ss = tt.splat %shift : i{bits} -> {typ}",
            f"%k = arith.constant dense<{self.stride}> : {typ}",
            f"%d = arith.constant dense<{d}> : {typ}",
            f"%pitch = arith.constant dense<{d+7}> : {typ}",
            f"%half = arith.constant dense<{max(1,d//2)}> : {typ}",
            f"%qbound = arith.constant dense<{max(1,b//d-1)}> : {typ}",
            f"%limit = tt.splat %length : i{bits} -> {typ}",
            f"%r32 = tt.make_range {{end = {b} : i32, start = 0 : i32}} : tensor<{b}xi32>",
        ]
        if bits==64: lines += [f"%r = arith.extsi %r32 : tensor<{b}xi32> to {typ}"]
        else: lines[-1]=lines[-1].replace("%r32","%r")
        if self.other=="tensor":
            lines += [f"%baseother = arith.constant dense<-17.0> : tensor<{b}xf32>",f"%rf = arith.sitofp %r : {typ} to tensor<{b}xf32>",f"%quarter = arith.constant dense<0.25> : tensor<{b}xf32>",f"%otheroffset = arith.mulf %rf, %quarter : tensor<{b}xf32>",f"%other = arith.addf %baseother, %otheroffset : tensor<{b}xf32>"]
        else: lines += [f"%other = arith.constant dense<{'0x7FC00000' if self.other=='nan' else '-17.0'}> : tensor<{b}xf32>"]
        lines += [
            f"%kr = arith.muli %r, %k : {typ}",
            f"%x = arith.addi %kr, %ss : {typ}",
            f"%rem = arith.remsi %x, %d : {typ}",
            f"%div = arith.divsi %x, %d : {typ}",
            f"%row = arith.muli %div, %pitch : {typ}",
            f"%packed = arith.addi %row, %rem : {typ}",
            f"%mlinear = arith.cmpi slt, %x, %limit : {typ}",
            f"%mperiodic = arith.cmpi slt, %rem, %half : {typ}",
            f"%mquotient = arith.cmpi slt, %div, %qbound : {typ}",
            f"%mboth = arith.andi %mlinear, %mperiodic : tensor<{b}xi1>",
            f"%ps = tt.splat %p : !tt.ptr<f32> -> tensor<{b}x!tt.ptr<f32>>",
            f"%ptr = tt.addptr %ps, %{dict(rem='rem',div='div',packed='packed')[self.operation]} : tensor<{b}x!tt.ptr<f32>>, {typ}",
        ]
        suffix="" if self.mask=="none" else f", %{dict(linear='mlinear',periodic='mperiodic',quotient='mquotient',combined='mboth')[self.mask]}, %other"
        lines += [f"%v = tt.load %ptr{suffix} : tensor<{b}x!tt.ptr<f32>>",f"tt.return %v : tensor<{b}xf32>","} }"]
        return "\n".join(lines)

def matrix(quick=False):
    divisors = [3, 16, 24, 65] if quick else [1,2,3,5,7,16,24,63,64,65,127,128,129,256]
    seen = set()
    for d in divisors:
        for s in sorted({0,1,d-1}):
            for op in ("rem","div","packed"):
                for mask in ("none","linear","periodic","quotient","combined"):
                    for dyn in (False,True):
                        c=Case(divisor=d,shift=s,operation=op,mask=mask,dynamic_shift=dyn)
                        if c not in seen: seen.add(c); yield c
    if not quick:
        for b in (64,256):
            for d in (3,16,65):
                for op in ("rem","div","packed"):
                    yield Case(block=b,divisor=d,operation=op)
        for k in (2,3,8,24):
            for d in (16,24):
                for s in (0,1):
                    for op in ("rem","div","packed"):
                        for mask in ("none","linear"):
                            yield Case(divisor=d,stride=k,shift=s,operation=op,mask=mask)

        for d in (3,16,65,256):
            for s in (0,1):
                for mask in ("none","linear","periodic","quotient","combined"):
                    for pid in (0,1,3):
                        yield Case(divisor=d,shift=s,operation="packed",mask=mask,program_shift=True,program_id=pid)
        for d in (3,16,65,256):
            for count in (0,1,127,128):
                yield Case(divisor=d,shift=1,operation="packed",mask="linear",valid_count=count)
            for other in ("nan","tensor"):
                yield Case(divisor=d,shift=1,operation="packed",mask="combined",other=other)
            yield Case(divisor=d,shift=1,operation="packed",mask="combined",bits=64)

        yield Case(block=64,divisor=3,shift=6,operation="rem",mask="combined",valid_count=0,other="nan")
        yield Case(block=64,divisor=2,shift=1,operation="rem",mask="combined",valid_count=1,bits=64)

def tensor_shape(text):
    matches = re.findall(r"tensor<([0-9?x]+)(?:f32|i[0-9]+|index|!tt\.ptr)",text)
    if not matches: return ()
    dims=matches[-1].rstrip("x").split("x")
    return tuple(-1 if d=="?" else int(d) for d in dims if d)

def scalar_token(token, env):
    token=token.strip()
    return int(np.asarray(env[token]).item()) if token.startswith("%") else int(token)

def slice_args(text,env):
    parts=re.findall(r"\[([^\]]*)\]",text)[:3]
    if len(parts)!=3: raise ValueError("cannot parse slice "+text)
    vec=[[scalar_token(t,env) for t in x.split(",") if t.strip()] for x in parts]
    return tuple(slice(o,o+n*s,s) for o,n,s in zip(*vec))

def numerical_eval(ir,case):
    """Small interpreter for side-effect-free address/shape IR emitted by this pass."""
    env={"%arg0":0,"%arg1":case.effective_shift+case.stride*case.count,"%arg2":case.effective_shift,
         "%p":0,"%length":case.effective_shift+case.stride*case.count,"%shift":case.effective_shift}
    addresses=[]
    lines=ir.splitlines()
    cursor=0
    while cursor<len(lines):
        line=lines[cursor].strip()
        cursor+=1
        if not line or line.startswith("//") or line in ("{","}"): continue
        if line.startswith("tt.return"):
            name=SSA.findall(line)[0]
            return np.asarray(env[name]), np.concatenate(addresses) if addresses else np.array([],dtype=int)
        conditional=re.match(r"(?:(%[-\w.$]+)\s*=\s*)?scf.if\s+(%[-\w.$]+)",line)
        if conditional:
            result,condition=conditional.groups()
            branches=[[],[]]
            which=0
            depth=1
            end=cursor
            while end<len(lines):
                nested=lines[end].strip()
                end+=1
                if depth==1 and nested.startswith("} else {"):
                    which=1
                    continue
                depth+=nested.count("{")-nested.count("}")
                if depth==0: break
                branches[which].append(lines[end-1])
            chosen=branches[0 if bool(np.asarray(env[condition]).item()) else 1]
            body=[]
            local_depth=0
            for statement in chosen:
                if local_depth==0 and statement.strip().startswith("scf.yield"):
                    if result:
                        refs_yield=SSA.findall(statement)
                        if len(refs_yield)!=1: raise NotImplementedError("multi-result scf.if")
                        body.append(result+" = evaluator.alias "+refs_yield[0])
                else: body.append(statement)
                local_depth+=statement.count("{")-statement.count("}")
            lines[cursor:end]=body
            continue
        match=re.match(r"(%[-\w.$]+)\s*=\s*([-\w.]+)\s*(.*)",line)
        if not match: continue
        name,op,rest=match.groups()
        refs=SSA.findall(rest)
        vals=[env[r] for r in refs if r in env]
        shape=tensor_shape(rest)
        if op=="evaluator.alias": env[name]=vals[0]
        elif op=="arith.constant":
            dense=re.search(r"dense<(.+?)>",rest)
            if dense:
                token=dense.group(1)
                if token.startswith("["):
                    value=np.asarray(json.loads(token.replace("true","1").replace("false","0")))
                    env[name]=value.reshape(shape)
                else:
                    value=np.array(int(token,16),dtype=np.uint32).view(np.float32).item() if token.startswith("0x") else (token=="true") if token in ("true","false") else float(token) if any(x in token.lower() for x in (".","e")) else int(token)
                    env[name]=np.full(shape,value)
            else:
                token=rest.split(":",1)[0].strip()
                env[name]=(token=="true") if token in ("true","false") else float(token) if "." in token or "e" in token.lower() else int(token)
        elif op=="tt.get_program_id": env[name]=case.program_id
        elif op=="tt.make_range":
            end=int(re.search(r"end\s*=\s*(\d+)",rest).group(1))
            start=int(re.search(r"start\s*=\s*(\d+)",rest).group(1))
            env[name]=np.arange(start,end,dtype=np.int64)
        elif op=="tt.splat": env[name]=np.full(shape,np.asarray(vals[0]).item())
        elif op=="tt.expand_dims":
            axis=int(re.search(r"axis\s*=\s*(\d+)",rest).group(1))
            env[name]=np.expand_dims(vals[0],axis)
        elif op=="tt.broadcast": env[name]=np.broadcast_to(vals[0],shape)
        elif op in ("arith.index_cast","arith.index_castui","arith.extsi","arith.extui","arith.trunci"): env[name]=np.asarray(vals[0]).astype(np.int64)
        elif op in ("arith.addi","tt.addptr"): env[name]=np.asarray(vals[0])+vals[1]
        elif op=="arith.sitofp": env[name]=np.asarray(vals[0]).astype(np.float32)
        elif op=="arith.addf": env[name]=np.asarray(vals[0])+vals[1]
        elif op=="arith.mulf": env[name]=np.asarray(vals[0])*vals[1]
        elif op=="arith.ceildivsi": env[name]=-np.floor_divide(-np.asarray(vals[0]),vals[1])
        elif op=="arith.floordivsi": env[name]=np.floor_divide(vals[0],vals[1])
        elif op=="arith.subi": env[name]=np.asarray(vals[0])-vals[1]
        elif op=="arith.muli": env[name]=np.asarray(vals[0])*vals[1]
        elif op in ("arith.divsi","arith.divui"): env[name]=np.trunc(np.asarray(vals[0])/vals[1]).astype(np.int64)
        elif op in ("arith.remsi","arith.remui"):
            env[name]=np.asarray(vals[0])-np.trunc(np.asarray(vals[0])/vals[1]).astype(np.int64)*vals[1]
        elif op in ("arith.minsi","arith.minui"): env[name]=np.minimum(vals[0],vals[1])
        elif op in ("arith.maxsi","arith.maxui"): env[name]=np.maximum(vals[0],vals[1])
        elif op=="arith.andi": env[name]=np.asarray(vals[0]) & vals[1]
        elif op=="arith.ori": env[name]=np.asarray(vals[0]) | vals[1]
        elif op=="arith.xori": env[name]=np.asarray(vals[0]) ^ vals[1]
        elif op=="arith.select": env[name]=np.where(vals[0],vals[1],vals[2])
        elif op=="arith.cmpi":
            pred=rest.split(",",1)[0].strip()
            fn={"slt":np.less,"ult":np.less,"sle":np.less_equal,"ule":np.less_equal,"sgt":np.greater,"ugt":np.greater,"sge":np.greater_equal,"uge":np.greater_equal,"eq":np.equal,"ne":np.not_equal}[pred]
            env[name]=fn(vals[0],vals[1])
        elif op=="tt.load":
            idx=np.asarray(vals[0],dtype=np.int64)
            mask=np.broadcast_to(vals[1],idx.shape).astype(bool) if len(vals)>1 else np.ones(idx.shape,dtype=bool)
            out=np.broadcast_to(vals[2],idx.shape).copy() if len(vals)>2 else np.full(idx.shape,np.nan,dtype=np.float64)
            active=idx[mask]
            if np.any(active<0) or np.any(active>=len(MEMORY)): raise AssertionError("load out of allocated range")
            out[mask]=MEMORY[active]
            env[name]=out
            addresses.append(active.reshape(-1))
        elif op=="tensor.empty": env[name]=np.full(shape,np.nan)
        elif op in ("tensor.reshape","tensor.collapse_shape","tensor.expand_shape"):
            if op=="tensor.reshape": target=tuple(int(x) for x in np.asarray(vals[1]).flat)
            else: target=shape
            env[name]=np.asarray(vals[0]).reshape(target)
        elif op=="tensor.extract_slice":
            env[name]=np.asarray(vals[0])[slice_args(rest,env)]
        elif op=="tensor.insert_slice":
            out=np.array(vals[1],copy=True)
            # Brackets start at the destination's offset list.
            out[slice_args(rest,env)]=vals[0]
            env[name]=out
        elif op=="linalg.broadcast":
            dims=[int(x) for x in re.search(r"dimensions\s*=\s*\[([^\]]*)\]",rest).group(1).split(",") if x.strip()]
            src=np.asarray(vals[0])
            for axis in sorted(dims): src=np.expand_dims(src,axis)
            env[name]=np.broadcast_to(src,np.asarray(vals[1]).shape)
        else: raise NotImplementedError(op)
    raise ValueError("no tt.return")

def pointer_is_structured(ir):
    defs={}
    ptrs=[]
    for line in ir.splitlines():
        m=re.match(r"\s*(%[-\w.$]+)\s*=\s*(.*)",line)
        if m:
            defs[m[1]]=m[2]
            if m[2].startswith("tt.load"):
                ptrs.append(SSA.findall(m[2])[0])
    visited=set()
    def visit(v):
        if v in visited: return False
        visited.add(v)
        expr=defs.get(v,"")
        if re.search(r"arith\.(?:div|rem)[su]i",expr) and "tensor<" in expr: return True
        return any(visit(x) for x in SSA.findall(expr))
    return not any(visit(p) for p in ptrs)

def run(opt,report,snapshots=None,quick=False,require_new=False):
    cases=list(matrix(quick))
    records=[]
    if snapshots: snapshots.mkdir(parents=True,exist_ok=True)
    for c in cases:
        r=subprocess.run([str(opt),"--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true"],input=c.mlir(),text=True,capture_output=True)
        row={"case":asdict(c),"name":c.name,"exit_code":r.returncode,"converted":r.returncode==0 and pointer_is_structured(r.stdout)}
        if r.returncode==0:
            try:
                got,actual_addr=numerical_eval(r.stdout,c)
                want,want_addr=c.expected()
                row["numerical_ok"]=got.shape==want.shape and np.array_equal(got,want,equal_nan=True)
                # Deduplicating valid loads is legal; addresses must be a subset of
                # source active addresses, since masked lanes may be out of bounds.
                row["address_safe"]=set(map(int,actual_addr)).issubset(set(map(int,want_addr)))
                if not row["numerical_ok"]: row["mismatch_lanes"]=np.flatnonzero(got!=want)[:12].tolist() if got.shape==want.shape else ["shape"]
            except Exception as exc:
                row["validation_error"]=f"{type(exc).__name__}: {exc}"
        else: row["stderr"]=r.stderr[:1500]
        if snapshots:
            (snapshots/(c.name+".input.mlir")).write_text(c.mlir())
            (snapshots/(c.name+".output.mlir")).write_text(r.stdout)
        records.append(row)
    summary={"cases":len(records),"converted":sum(x["converted"] for x in records),"numerically_validated":sum("numerical_ok" in x for x in records),"numerical_failures":sum(x.get("numerical_ok") is False for x in records),"unsafe_address_sets":sum(x.get("address_safe") is False for x in records),"unsupported_evaluator_cases":sum("validation_error" in x for x in records),"nonzero_exit":sum(x["exit_code"]!=0 for x in records)}
    data={"triton_opt":str(opt),"summary":summary,"records":records}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(data,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if require_new:
        failed=[x for x in records if not x["converted"] or x.get("numerical_ok") is not True or x.get("address_safe") is not True]
        if failed: print("Failing cases:",[x["name"] for x in failed[:20]]); return 1
    return 0

def test_empty_memory_is_eliminated_success():
    assert pointer_is_structured("tt.return %other : tensor<128xf32>")

def test_phase_reference():
    c=Case(shift=1)
    expected,_=c.expected()
    assert expected[15]==MEMORY[0] and expected[0]==MEMORY[1]

def test_matrix_unique():
    rows=list(matrix())
    assert len({x.name for x in rows})==len(rows)

def test_interpreter_conditional_scalar_load():
    ir="""module { tt.func @test(%arg0: !tt.ptr<f32>) -> tensor<128xf32> {
%c1 = arith.constant 1 : i32
%c2 = arith.constant 2 : i32
%cBad = arith.constant 99999999 : i32
%condition = arith.cmpi slt, %c1, %c2 : i32
%value = scf.if %condition -> (i32) {
  %inner = scf.if %condition -> (i32) {
    scf.yield %c1 : i32
  } else {
    scf.yield %cBad : i32
  }
  scf.yield %inner : i32
} else {
  scf.yield %cBad : i32
}
%ptr = tt.addptr %arg0, %value : !tt.ptr<f32>, i32
%v = tt.load %ptr : !tt.ptr<f32>
%out = tt.splat %v : f32 -> tensor<128xf32>
tt.return %out : tensor<128xf32>
} }"""
    result,addresses=numerical_eval(ir,Case())
    assert addresses.tolist()==[1]
    assert np.all(result==MEMORY[1])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--triton-opt",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--snapshots",type=Path)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--require-new",action="store_true",help="Require every case to convert, execute correctly, and preserve active addresses")
    a=p.parse_args()
    sys.exit(run(a.triton_opt,a.report,a.snapshots,a.quick,a.require_new))
