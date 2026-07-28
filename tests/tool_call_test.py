import json, os, urllib.request, sys

URL=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1/chat/completions"
MODEL=os.environ.get("MODEL", "GLM-5.2-EXL3-TR3-3.0bpw")
TOOLS=[{"type":"function","function":{
  "name":"get_case_status",
  "description":"Look up the current status of a Kentucky court case by case number.",
  "parameters":{"type":"object","properties":{
    "case_number":{"type":"string","description":"e.g. 24-CI-00123"},
    "county":{"type":"string","description":"Kentucky county name"}},
    "required":["case_number","county"]}}},
 {"type":"function","function":{
  "name":"compute_deadline",
  "description":"Compute a filing deadline given a trigger date and rule days.",
  "parameters":{"type":"object","properties":{
    "trigger_date":{"type":"string"},"days":{"type":"integer"}},
    "required":["trigger_date","days"]}}}]

def ask(messages, tools=None, tool_choice=None, max_tokens=4096):
    body={"model":MODEL,"messages":messages,"temperature":0,"max_tokens":max_tokens}
    if tools: body["tools"]=tools
    if tool_choice: body["tool_choice"]=tool_choice
    req=urllib.request.Request(URL,method="POST",headers={"Content-Type":"application/json"},data=json.dumps(body).encode())
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())

fails=0
# T1: should call get_case_status with parsed args
d=ask([{"role":"user","content":"What's the status of case 24-CI-00123 in Carter county?"}],tools=TOOLS)
ch=d["choices"][0]; tc=(ch["message"].get("tool_calls") or [])
print(f"T1 auto-call: finish={ch['finish_reason']} tool_calls={len(tc)}")
if tc:
    f0=tc[0]["function"]; args=json.loads(f0["arguments"])
    ok = f0["name"]=="get_case_status" and args.get("case_number")=="24-CI-00123" and "carter" in str(args.get("county","")).lower()
    print(f"   name={f0['name']} args={args} -> {'PASS' if ok else 'FAIL'}"); fails += 0 if ok else 1
else: print("   FAIL: no tool call emitted"); fails+=1

# T2: negative — plain question must NOT force a bogus call
d=ask([{"role":"user","content":"In one word, what color is the sky on a clear day?"}],tools=TOOLS)
ch=d["choices"][0]; tc=(ch["message"].get("tool_calls") or [])
ok = not tc and "blue" in (ch["message"].get("content") or "").lower()
print(f"T2 no-spurious-call: tool_calls={len(tc)} content~blue={ok} -> {'PASS' if ok else 'FAIL'}"); fails += 0 if ok else 1

# T3: full round trip — feed tool result back, expect grounded final answer
msgs=[{"role":"user","content":"What's the status of case 24-CI-00123 in Carter county?"}]
d=ask(msgs,tools=TOOLS); m=d["choices"][0]["message"]
if m.get("tool_calls"):
    msgs.append({"role":"assistant","content":m.get("content") or "","tool_calls":m["tool_calls"]})
    msgs.append({"role":"tool","tool_call_id":m["tool_calls"][0]["id"],"content":json.dumps({"status":"Active — pretrial conference set 2026-08-14"})})
    d2=ask(msgs,tools=TOOLS); final=(d2["choices"][0]["message"].get("content") or "")
    ok = "2026-08-14" in final or "pretrial" in final.lower()
    print(f"T3 round-trip: final uses tool result -> {'PASS' if ok else 'FAIL'} ({final.strip()[:70]!r})"); fails += 0 if ok else 1
else: print("T3 FAIL: no call to round-trip"); fails+=1

# T4: tool_choice=required with two tools — must pick one and emit valid JSON
d=ask([{"role":"user","content":"If an order was entered 2026-07-20 and I have 30 days to appeal, when is my deadline?"}],tools=TOOLS,tool_choice="required")
ch=d["choices"][0]; tc=(ch["message"].get("tool_calls") or [])
if tc:
    f0=tc[0]["function"]
    try: args=json.loads(f0["arguments"]); ok=f0["name"]=="compute_deadline" and args.get("days")==30
    except Exception: ok=False; args="<unparseable>"
    print(f"T4 required: name={f0['name']} args={args} -> {'PASS' if ok else 'FAIL'}"); fails += 0 if ok else 1
else: print("T4 FAIL: required but no tool call"); fails+=1

print(f"\nRESULT: {4-fails}/4 tool-call tests passed")
sys.exit(1 if fails else 0)
