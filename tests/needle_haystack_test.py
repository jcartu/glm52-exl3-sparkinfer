#!/usr/bin/env python3
"""
Needle-in-a-haystack probe for the EXL3 + MTP-3 production server.

Purpose: test the regime vLLM issue #183 claims is corrupted by
VLLM_EXL3_TRELLIS_MIN_M=1 -- long INPUT context. The existing estonia profile
does not cover this: its 40000 is a max-OUTPUT budget for a reasoning task, not
a long prompt.

Plants a unique fact at several depths in a long filler document, then asks for
it back with greedy decoding. A miss at long context with hits at short context
is the corruption signature described in the issue.
"""
import argparse, json, os, random, sys, time
import urllib.request

FILLER = (
    "The quarterly logistics review noted routine variance in regional throughput. "
    "Warehouse utilization tracked close to forecast, and carrier performance held "
    "within contractual tolerances. No exceptions were escalated to the steering group. "
)

def build_prompt(ctx_tokens, depth_frac, secret, chars_per_token=6.63):
    total_chars = int(ctx_tokens * chars_per_token)
    reps = max(1, total_chars // len(FILLER))
    body = FILLER * reps
    needle = (f"\n\nIMPORTANT RECORD: the authorization code for the Rhinebeck "
              f"transfer is {secret}.\n\n")
    cut = int(len(body) * depth_frac)
    return (body[:cut] + needle + body[cut:] +
            "\n\nQuestion: what is the authorization code for the Rhinebeck transfer? "
            "Reply with the code only.")

def ask(url, model, prompt, max_tokens=None, timeout=1800):
    req = urllib.request.Request(
        url, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **({"max_tokens": max_tokens} if max_tokens else {}),
            "temperature": 0,
        }).encode())
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    txt = (msg.get("content") or "")
    return txt, d.get("usage", {}), time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1/chat/completions")
    ap.add_argument("--model", default=os.environ.get("MODEL", "GLM-5.2-EXL3-TR3-3.0bpw"))
    ap.add_argument("--contexts", default="8000,16000,30000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rng = random.Random(20260726)
    rows, fails = [], 0
    for ctx in [int(c) for c in a.contexts.split(",")]:
        for depth in [float(d) for d in a.depths.split(",")]:
            secret = f"{rng.randint(10**7, 10**8-1)}"
            prompt = build_prompt(ctx, depth, secret)
            try:
                txt, usage, dt = ask(a.url, a.model, prompt)
                hit = secret in txt
                # NaN/garbage signature: empty or non-ascii soup
                garbled = (not txt.strip()) or (sum(c.isprintable() for c in txt) < len(txt) * 0.8)
                rows.append({"ctx_target": ctx, "depth": depth, "hit": hit,
                             "garbled": garbled, "prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens"),
                             "seconds": round(dt, 1), "reply": txt.strip()[:80]})
                if not hit:
                    fails += 1
                flag = "HIT " if hit else "MISS"
                if garbled:
                    flag = "GARBLED"
                print(f"  ctx~{ctx:>7} depth {depth:>4} prompt_tokens={usage.get('prompt_tokens')} "
                      f"{flag} ({dt:.0f}s) reply={txt.strip()[:40]!r}", flush=True)
            except Exception as e:
                fails += 1
                rows.append({"ctx_target": ctx, "depth": depth, "error": str(e)[:160]})
                print(f"  ctx~{ctx:>7} depth {depth:>4} ERROR {str(e)[:110]}", flush=True)

    total = len(rows)
    print(f"\n  RESULT: {total-fails}/{total} needles recovered")
    if a.out:
        json.dump({"rows": rows, "passed": total - fails, "total": total},
                  open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
