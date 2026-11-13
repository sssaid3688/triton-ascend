"""Seeded end-to-end IR/address tests for supported segmented domains."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import subprocess
import numpy as np
import pytest
from test_segmented_linearization import Case, numerical_eval, pointer_is_structured



class IntervalCase(Case):
    @property
    def name(self):
        return super().name + "_lower_intervals"

    def mlir(self):
        text=super().mlir()
        t=f"tensor<{self.block}xi{self.bits}>"
        old=f"%mboth = arith.andi %mlinear, %mperiodic : tensor<{self.block}xi1>"
        new=f"""%lowx = arith.constant dense<{self.shift+5}> : {t}
%lowr = arith.constant dense<1> : {t}
%mlowx = arith.cmpi sge, %x, %lowx : {t}
%mlowr = arith.cmpi sge, %rem, %lowr : {t}
%mlow = arith.andi %mlowx, %mlowr : tensor<{self.block}xi1>
%mhigh = arith.andi %mlinear, %mperiodic : tensor<{self.block}xi1>
%mboth = arith.andi %mlow, %mhigh : tensor<{self.block}xi1>"""
        assert old in text
        return text.replace(old,new)

    def expected(self):
        result,_=super().expected()
        x=self.shift+np.arange(self.block,dtype=np.int64)
        legal=(x>=self.shift+5)&(x<self.shift+self.count)&(x%self.divisor>=1)&(x%self.divisor<max(1,self.divisor//2))
        other=(-17+np.arange(self.block,dtype=np.float32)*.25) if self.other=="tensor" else np.full(self.block,np.nan if self.other=="nan" else -17,dtype=np.float32)
        result[~legal]=other[~legal]
        r,q=x%self.divisor,x//self.divisor
        idx=r if self.operation=="rem" else q if self.operation=="div" else q*(self.divisor+7)+r
        return result,idx[legal]


def cases():
    rng = random.Random(20261113)
    for _ in range(256):
        b = rng.choice([16, 32, 64, 128, 256, 512])
        d = rng.choice([1, 2, 3, 5, 7, 13, 16, 24, 30, 31, 63, 65, 127, 129, 257])
        yield Case(block=b, divisor=d, shift=rng.randrange(3*d),
                   operation=rng.choice(["rem", "div", "packed"]),
                   mask=rng.choice(["none", "linear", "periodic", "combined", "quotient"]),
                   program_shift=rng.choice([False, True]), program_id=rng.randrange(5),
                   other=rng.choice(["scalar", "tensor", "nan"]),
                   valid_count=rng.choice([0, 1, b-1, b, b+1]), bits=rng.choice([32,64]))


    for d in [3, 7, 16, 65]:
        for shift in [0, 1]:
            for operation in ["rem", "div", "packed"]:
                for other in ["tensor", "nan"]:
                    yield IntervalCase(block=128,divisor=d,shift=shift,operation=operation,
                                       mask="combined",other=other)


def evaluate(opt, case, fallback=False):
    result = subprocess.run([str(opt), f"--triton-to-structured=enable-mask-fallback-conversion={str(fallback).lower()} optimize-dynamic-offset=true"],
                            input=case.mlir(), capture_output=True, text=True)
    record = {"case": asdict(case), "exit_code": result.returncode}
    if result.returncode:
        record["error"] = result.stderr[:2000]
        return record
    record["converted"] = pointer_is_structured(result.stdout)
    try:
        actual, addresses = numerical_eval(result.stdout, case)
        expected, legal = case.expected()
        record["numerical_ok"] = actual.shape == expected.shape and np.array_equal(actual,expected,equal_nan=True)
        record["address_safe"] = set(map(int,addresses)).issubset(set(map(int,legal)))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


@pytest.mark.parametrize("case", list(cases()), ids=lambda c:c.name)
def test_seeded_segmented_ir(case):
    opt = shutil.which("triton-opt")
    if opt is None:
        pytest.skip("triton-opt build required")
    row = evaluate(opt,case)
    assert row.get("converted") and row.get("numerical_ok") and row.get("address_safe"), row


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--triton-opt", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mask-fallback", action="store_true")
    args=parser.parse_args()
    records=[evaluate(args.triton_opt,c,args.mask_fallback) for c in cases()]
    failures=[r for r in records if not (r.get("converted") and r.get("numerical_ok") and r.get("address_safe"))]
    data={"seed":20261113,"cases":len(records),"failures":len(failures),"records":records}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!="records"}))
    for failure in failures[:6]: print(failure)
    raise SystemExit(bool(failures))
