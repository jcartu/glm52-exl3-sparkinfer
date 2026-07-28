import json, os, urllib.request
URL=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1/chat/completions"
MODEL=os.environ.get("MODEL", "GLM-5.2-EXL3-TR3-3.0bpw")
TOOLS=[{"type":"function","function":{"name":"read_file","description":"Read a file",
  "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
body={"model":MODEL,"temperature":0,"stream":True,
  "tools":TOOLS,
  "messages":[{"role":"user","content":"Read the file README.md using the tool."}]}
req=urllib.request.Request(URL,method="POST",headers={"Content-Type":"application/json"},data=json.dumps(body).encode())
name_parts=[]; arg_parts=[]; content_parts=[]; finish=None; chunks=0
with urllib.request.urlopen(req,timeout=300) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data: ") or line=="data: [DONE]": continue
        chunks+=1
        d=json.loads(line[6:])
        ch=d["choices"][0]; delta=ch.get("delta",{})
        if ch.get("finish_reason"): finish=ch["finish_reason"]
        if delta.get("content"): content_parts.append(delta["content"])
        for tc in (delta.get("tool_calls") or []):
            f=tc.get("function",{})
            if f.get("name"): name_parts.append(f["name"])
            if f.get("arguments"): arg_parts.append(f["arguments"])
name="".join(name_parts); args="".join(arg_parts); content="".join(content_parts)
print(f"chunks={chunks} finish={finish}")
print(f"streamed tool name: {name!r}")
print(f"streamed args     : {args!r}")
try:
    ok = name=="read_file" and json.loads(args).get("path","").lower().endswith("readme.md")
except Exception as e:
    ok=False; print(f"args JSON parse FAILED: {e}")
print(f"leaked content text: {content!r}" if content else "no content leak")
print("STREAMING TOOL CALL:", "PASS" if ok and finish=="tool_calls" else "FAIL")
