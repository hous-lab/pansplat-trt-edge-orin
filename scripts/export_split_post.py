#!/usr/bin/env python
"""Split cv_post1/cv_post2 ONNX into 4 segments around attention blocks."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

BLOCK_SUF = "/attention/Transpose_1"


def infer(path):
    m = onnx.load(path)
    return shape_inference.infer_shapes(m)


def shapes_of(m):
    out = {}
    for v in list(m.graph.value_info) + list(m.graph.input) \
            + list(m.graph.output):
        t = v.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        out[v.name] = (t.elem_type, dims)
    return out


def find_blocks(g):
    return [n.name[:-len(BLOCK_SUF)]
            for n in g.node if n.name.endswith(BLOCK_SUF)]


def make_cast(name_in, name_out, to_fp16, tag):
    to = TensorProto.FLOAT16 if to_fp16 else TensorProto.FLOAT
    return helper.make_node("Cast", [name_in], [name_out], to=to, name=tag)


def build_cut_tensors(pfx, blocks):
    """Per block: (q,k_src,v,o fp32 names, q16/k16/v16/o16 cut names)."""
    cuts = []
    for bi, blk in enumerate(blocks):
        n = bi + 1
        q = blk + "/attention/Transpose_1_output_0"
        v = blk + "/attention/Transpose_2_output_0"
        ks = blk + "/attention/Split_output_1"
        o = blk + "/attention/MatMul_1_output_0"
        kt = blk + "/attention/k_t"
        cuts.append({
            "blk": blk, "q": q, "k_src": ks, "v": v, "o": o,
            "q16": "%s_q%d" % (pfx, n), "k16": "%s_k%d" % (pfx, n),
            "v16": "%s_v%d" % (pfx, n), "o16": "%s_o%d" % (pfx, n),
            "kt": kt,
        })
    return cuts




def backreach(prod, outs, avail, banned, nodes):
    need = set()
    stack = list(outs)
    while stack:
        t = stack.pop()
        if t == "":
            continue
        if t in avail or t in banned:
            continue
        i = prod[t]
        if i in need:
            continue
        need.add(i)
        stack.extend(nodes[i].input)
    return need



def extended_graph(g, cuts, no_cast=False):
    """Append k-transpose + boundary casts; return extended node list.

    no_cast=True: boundary tensors stay fp32 (bit-exact verify mode)."""
    ext = list(g.node)
    for c in cuts:
        ext.append(helper.make_node(
            "Transpose", [c["k_src"]], [c["kt"]], perm=[0, 2, 1],
            name=c["blk"] + "/attention/k_t"))
        if no_cast:
            continue
        ext.append(make_cast(c["q"], c["q16"], True, c["q16"] + "_cast"))
        ext.append(make_cast(c["kt"], c["k16"], True, c["k16"] + "_cast"))
        ext.append(make_cast(c["v"], c["v16"], True, c["v16"] + "_cast"))
        ext.append(make_cast(c["o16"], c["o"], False, c["o16"] + "_cast"))
    return ext




def true_shapes(src, tensors):
    """True runtime dims via one extract_model + ORT run (ONNX shape
    inference returns 0-dims on the Resize path, so we ask ORT instead)."""
    import onnxruntime as ort
    if not tensors:
        return {}
    tmp = str(Path(__file__).resolve().parent.parent / "export" / "scratch" / "flash_shape_probe.onnx")
    m = onnx.load(src)
    gin = [i.name for i in m.graph.input]
    onnx.utils.extract_model(src, tmp, gin, tensors)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        dims = [d if isinstance(d, int) else 1 for d in i.shape]
        feeds[i.name] = np.random.rand(*dims).astype(np.float32)
    outs = sess.run(tensors, feeds)
    return {t: [int(d) for d in o.shape] for t, o in zip(tensors, outs)}


def split_engine(src, outdir, pfx, no_cast=False):
    model = infer(src)
    g = model.graph
    nodes = list(g.node)
    blocks = find_blocks(g)
    print("%s: %d nodes" % (src, len(nodes)))
    assert len(blocks) == 3
    shapes = shapes_of(model)
    cuts = build_cut_tensors(pfx, blocks)
    prod = {}
    cons = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            prod[o] = i
        for t in n.input:
            if t != "":
                cons.setdefault(t, []).append(i)
    inits = set(t.name for t in g.initializer)
    gin = set(i.name for i in g.input)
    gout = [o.name for o in g.output]
    banned = set()
    for c in cuts:
        banned |= backreach(prod, [c["o"]],
                            set([c["q"], c["k_src"], c["v"]]), set(), nodes)
    ext = extended_graph(g, cuts, no_cast)
    ext_prod = {}
    for i, n in enumerate(ext):
        for o in n.output:
            ext_prod[o] = i
    kb = "kt" if no_cast else "k16"
    qb = "q" if no_cast else "q16"
    vb = "v" if no_cast else "v16"
    ob = "o" if no_cast else "o16"
    seg_want = [[cuts[s][qb], cuts[s][kb], cuts[s][vb]]
                for s in range(3)] + [gout]
    seg_avail_extra = [set(), {cuts[0][ob]}, {cuts[1][ob]}, {cuts[2][ob]}]
    avail = gin | inits
    segs = []
    for s in range(4):
        need = backreach(ext_prod, seg_want[s], avail | seg_avail_extra[s],
                         banned, ext)
        segs.append(need)
        if s < 3:
            for i in need:
                avail |= set(ext[i].output)
            avail |= {cuts[s]["o16"]}
    for s in range(4):
        assert not (segs[s] & banned)
    assigned = set()
    for s in range(4):
        assigned |= segs[s]
    leftover = set(range(len(ext))) - assigned - banned
    if leftover:
        bad = [ext[i].name for i in sorted(leftover)]
        raise SystemExit("unassigned nodes: %d e.g. %s" % (len(leftover), bad[:5]))
    print("  dropped=%d  seg nodes=%s" % (len(banned), [len(s) for s in segs]))
    return model, g, nodes, ext, cuts, shapes, inits, gin, gout, segs, banned


def write_segments(src, outdir, pfx, no_cast=False):
    (model, g, nodes, ext, cuts, shapes, inits, gin, gout,
     segs, banned) = split_engine(src, outdir, pfx, no_cast)
    tag = ["a", "m1", "m2", "b"]
    ext_prod = {}
    for i, n in enumerate(ext):
        for o in n.output:
            ext_prod[o] = i
    ext_cons = {}
    for i, n in enumerate(ext):
        for t in n.input:
            if t != "":
                ext_cons.setdefault(t, []).append(i)
    seg_ins, seg_outs = [], []
    for s in range(4):
        kept = segs[s]
        produced = set()
        for i in kept:
            produced |= set(ext[i].output)
        consumed = set()
        for i in kept:
            consumed |= set(t for t in ext[i].input if t)
        ins = consumed - produced - inits
        live = lambda t: [j for j in ext_cons.get(t, []) if j not in banned]
        outs = set(t for t in produced
                   if any(j not in kept for j in live(t)) or not live(t))
        seg_ins.append(ins)
        seg_outs.append(outs)
    # ---- resolve shapes/dtypes for all boundary tensors ----
    qkv = {"q16": "q", "k16": "k_src", "v16": "v", "o16": "o",
           "kt": "k_src", "q": "q", "v": "v", "o": "o"}
    cut_keys = ["q16", "k16", "v16", "o16"] if not no_cast else \
        ["q", "kt", "v", "o"]

    def cut_of(t):
        for c in cuts:
            for key in cut_keys:
                if c[key] == t:
                    return c, key
        return None, None

    cut_dims = {}
    for c in cuts:
        cut_dims[c["q16"]] = shapes[c["q"]][1]
        cut_dims[c["k16"]] = shapes[c["k_src"]][1]
        cut_dims[c["v16"]] = shapes[c["v"]][1]
        cut_dims[c["o16"]] = shapes[c["o"]][1]
    # boundary tensor -> (et, dims); 0-dims fixed via ORT oracle
    bnd = set()
    for s in range(4):
        bnd |= seg_ins[s] | seg_outs[s]
    tainted = [t for t in sorted(bnd)
               if t not in gin and not cut_of(t)[0]
               and any(d == 0 for d in shapes[t][1])]
    oracle = true_shapes(src, tainted) if tainted else {}
    if tainted:
        print("  fixed 0-dim boundary tensors via ORT: %d" % len(tainted))
    bshape = {}

    def resolve(t):
        if t in gin:
            gvi = [i for i in g.input if i.name == t][0]
            tt = gvi.type.tensor_type
            return tt.elem_type, [d.dim_value for d in tt.shape.dim]
        c, key = cut_of(t)
        if c:
            et16 = TensorProto.FLOAT16 if not no_cast else TensorProto.FLOAT
            src_key = "q" if key == "kt" else qkv[key]
            return et16, shapes[c[src_key]][1]
        if t in oracle:
            return 1, oracle[t]
        return shapes[t]
    # ---- build + write the 4 subgraphs ----
    tag = ["a", "m1", "m2", "b"]
    meta = {}
    opset = model.opset_import[0].version
    for s in range(4):
        ins, outs = seg_ins[s], seg_outs[s]
        kept = segs[s]
        avail = ins | inits
        order = []
        placed = set()
        remaining = set(kept)
        while remaining:
            prog = False
            for i in sorted(remaining):
                if all(t == "" or t in avail or t in placed
                       for t in ext[i].input):
                    order.append(i)
                    placed |= set(ext[i].output)
                    remaining.discard(i)
                    prog = True
            if not prog:
                raise SystemExit("cycle inside segment %d" % s)
        sub_nodes = [ext[i] for i in order]
        used_inits = [t for t in g.initializer
                      if any(t.name == x for n in sub_nodes for x in n.input)]
        gin_names = set(i.name for i in g.input)
        sub_in = []
        for t in sorted(ins):
            et, dims = resolve(t)
            sub_in.append(helper.make_tensor_value_info(t, et, dims))
        sub_out = []
        for t in sorted(outs):
            et, dims = resolve(t)
            sub_out.append(helper.make_tensor_value_info(t, et, dims))
        graph = helper.make_graph(
            sub_nodes, "%s_%s" % (pfx, tag[s]), sub_in, sub_out,
            initializer=used_inits)
        m2 = helper.make_model(graph, opset_imports=model.opset_import,
                               producer_name="split_post")
        m2.ir_version = model.ir_version
        onnx.save(m2, os.path.join(outdir, "%s_%s.onnx" % (pfx, tag[s])))
        meta["%s_%s" % (pfx, tag[s])] = {
            "inputs": [[t, resolve(t)[0], resolve(t)[1]] for t in sorted(ins)],
            "outputs": [[t, resolve(t)[0], resolve(t)[1]] for t in sorted(outs)],
            "nodes": len(sub_nodes)}
    with open(os.path.join(outdir, "%s_split_meta.json" % pfx), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote %s_{a,m1,m2,b}.onnx + split_meta.json to %s" % (pfx, outdir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("pfx")
    ap.add_argument("--no-cast", action="store_true",
                    help="boundary tensors stay fp32 (bit-exact verify)")
    args = ap.parse_args()
    write_segments(args.src, args.outdir, args.pfx, args.no_cast)


if __name__ == "__main__":
    main()
