import os, sys, json, socket, argparse, requests
from typing import Dict, Any, List

POLLINATIONS_BASE_URL = os.getenv("POLLINATIONS_BASE_URL", "https://text.pollinations.ai/openai")
POLLINATIONS_MODEL = os.getenv("POLLINATIONS_MODEL", "openai")
POLLINATIONS_TOKEN = os.getenv("POLLINATIONS_TOKEN", "")

BLENDER_HOST = os.getenv("BLENDER_HOST", "localhost")
BLENDER_PORT = int(os.getenv("BLENDER_PORT", "9876"))

REQUEST_TIMEOUT = 120

class BlenderConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        if self.sock:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        self.sock = s

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.sock:
            self.connect()
        cmd = {"type": command_type, "params": params or {}}
        self.sock.sendall(json.dumps(cmd).encode("utf-8"))

        chunks: List[bytes] = []
        self.sock.settimeout(30.0)
        while True:
            chunk = self.sock.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
            try:
                merged = b"".join(chunks).decode("utf-8")
                obj = json.loads(merged)
                if isinstance(obj, dict):
                    break
            except json.JSONDecodeError:
                continue

        if not chunks:
            raise RuntimeError("No data received from Blender add-on")
        resp = json.loads(b"".join(chunks).decode("utf-8"))
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("message", "Blender error"))
        return resp.get("result", resp)

TOOLS = [
    {"type":"function","function":{"name":"get_scene_info","description":"Get Blender scene details.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_object_info","description":"Get object info by name.","parameters":{"type":"object","properties":{"object_name":{"type":"string"}},"required":["object_name"]}}},
    {"type":"function","function":{"name":"get_viewport_screenshot","description":"Capture viewport screenshot (returns file path + size).","parameters":{"type":"object","properties":{"max_size":{"type":"integer","default":800}}}}},
    {"type":"function","function":{"name":"execute_code","description":"Execute Python in Blender.","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}},
    {"type":"function","function":{"name":"get_polyhaven_status","description":"Check PolyHaven status.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_hyper3d_status","description":"Check Hyper3D Rodin status.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_sketchfab_status","description":"Check Sketchfab status.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_polyhaven_categories","description":"List PolyHaven categories.","parameters":{"type":"object","properties":{"asset_type":{"type":"string","enum":["hdris","textures","models","all"],"default":"hdris"}}}}},
    {"type":"function","function":{"name":"search_polyhaven_assets","description":"Search PolyHaven assets.","parameters":{"type":"object","properties":{"asset_type":{"type":"string","enum":["hdris","textures","models","all"],"default":"all"},"categories":{"type":"string"}}}}},
    {"type":"function","function":{"name":"download_polyhaven_asset","description":"Download/import PolyHaven asset.","parameters":{"type":"object","properties":{"asset_id":{"type":"string"},"asset_type":{"type":"string","enum":["hdris","textures","models"]},"resolution":{"type":"string","default":"1k"},"file_format":{"type":"string"}},"required":["asset_id","asset_type"]}}},
    {"type":"function","function":{"name":"set_texture","description":"Apply downloaded PolyHaven texture to object.","parameters":{"type":"object","properties":{"object_name":{"type":"string"},"texture_id":{"type":"string"}},"required":["object_name","texture_id"]}}},
    {"type":"function","function":{"name":"search_sketchfab_models","description":"Search Sketchfab.","parameters":{"type":"object","properties":{"query":{"type":"string"},"categories":{"type":"string"},"count":{"type":"integer","default":20},"downloadable":{"type":"boolean","default":True}},"required":["query"]}}},
    {"type":"function","function":{"name":"download_sketchfab_model","description":"Download/import Sketchfab model by UID.","parameters":{"type":"object","properties":{"uid":{"type":"string"}},"required":["uid"]}}},
    {"type":"function","function":{"name":"generate_hyper3d_model_via_text","description":"Generate 3D via Hyper3D text prompt.","parameters":{"type":"object","properties":{"text_prompt":{"type":"string"},"bbox_condition":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}},"required":["text_prompt"]}}},
    {"type":"function","function":{"name":"generate_hyper3d_model_via_images","description":"Generate 3D via Hyper3D images.","parameters":{"type":"object","properties":{"input_image_paths":{"type":"array","items":{"type":"string"}},"input_image_urls":{"type":"array","items":{"type":"string"}},"bbox_condition":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}}}}},
    {"type":"function","function":{"name":"poll_rodin_job_status","description":"Poll Hyper3D task.","parameters":{"type":"object","properties":{"subscription_key":{"type":"string"},"request_id":{"type":"string"}}}}},
    {"type":"function","function":{"name":"import_generated_asset","description":"Import generated Hyper3D asset.","parameters":{"type":"object","properties":{"name":{"type":"string"},"task_uuid":{"type":"string"},"request_id":{"type":"string"}},"required":["name"]}}},
]

ASSET_STRATEGY = """When creating Blender content:
0) Always start by calling get_scene_info.
1) Check integrations via get_polyhaven_status, get_sketchfab_status, get_hyper3d_status.
   Use PolyHaven for HDRIs/textures/generic models; Sketchfab for specific realistic models; Hyper3D for custom single items.
2) For Hyper3D: generate -> poll -> import -> adjust transforms. Do not try to generate whole scenes.
3) Verify world_bounding_box relationships; avoid clipping.
4) Prefer libraries (Sketchfab/PolyHaven) before scripting; use execute_code only when needed.
Return raw summaries and IDs clearly. Avoid fabrication; call tools whenever in doubt."""

def openai_headers():
    h = {"Content-Type": "application/json"}
    if POLLINATIONS_TOKEN:
        h["Authorization"] = f"Bearer {POLLINATIONS_TOKEN}"
    return h

def chat_completion(messages, tools=None):
    url = f"{POLLINATIONS_BASE_URL}/v1/chat/completions"
    payload = {
        "model": POLLINATIONS_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = requests.post(url, headers=openai_headers(), data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def run_with_tools(user_prompt: str):
    sys_msg = {"role":"system","content":"You are Pollinations Blender Agent. Use tools to query Blender and asset APIs. Be concise. If a tool exists, call it rather than guessing.\n" + ASSET_STRATEGY}
    msgs: List[Dict[str, Any]] = [sys_msg, {"role":"user","content":user_prompt}]
    bc = BlenderConnection(BLENDER_HOST, BLENDER_PORT)
    try:
        while True:
            resp = chat_completion(msgs, tools=TOOLS)
            choice = resp["choices"][0]["message"]
            tool_calls = choice.get("tool_calls")
            if not tool_calls:
                return choice.get("content","").strip()
            msgs.append({"role":"assistant","content":choice.get("content") or "", "tool_calls":tool_calls})
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"].get("arguments") or "{}"
                try:
                    parsed = json.loads(args)
                except Exception:
                    parsed = {}
                try:
                    result = bc.send_command(name, parsed)
                    content = json.dumps(result)
                except Exception as e:
                    content = json.dumps({"error": str(e)})
                msgs.append({"role":"tool","tool_call_id":tc["id"],"name":name,"content":content})
    finally:
        bc.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--message", help="Send a single prompt and print the answer.")
    args = ap.parse_args()
    if args.message:
        out = run_with_tools(args.message)
        print(out)
        return
    print("Pollinations Blender Agent. Type your prompt. Ctrl+C to exit.")
    try:
        while True:
            user = input("> ").strip()
            if not user:
                continue
            out = run_with_tools(user)
            print(out)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
