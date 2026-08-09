from __future__ import annotations
PROGRAMS = ["previous_token","induction","positional_decay","delimiter","bos_sink","uniform"]

def enrich_items(items, cfg):
    out=[]
    for i,row in enumerate(items):
        r=dict(row)
        r["program"]=PROGRAMS[i%len(PROGRAMS)]
        r["head_id"]=i%12
        r["layer_id"]=i%6  # valid for gpt2
        out.append(r)
    return out

