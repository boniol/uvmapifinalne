from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os, shutil
import urllib.request
from urllib.request import Request
from inference import separate_vocals

app = FastAPI()

MODEL_URL  = "https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_PATH = "models/UVR-MDX-NET-Inst_HQ_3.onnx"

@app.on_event("startup")
def download_model():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model…")
        req = Request(MODEL_URL)
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as resp, open(MODEL_PATH, "wb") as out:
            out.write(resp.read())
        print("✅ Model ready.")

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    in_dir  = "input_audio";  out_dir = "output_audio"
    os.makedirs(in_dir, exist_ok=True);  os.makedirs(out_dir, exist_ok=True)
    in_path  = f"{in_dir}/{file.filename}"
    out_path = f"{out_dir}/{file.filename}_inst.wav"
    with open(in_path, "wb") as buf:  shutil.copyfileobj(file.file, buf)
    separate_vocals(in_path, out_path)
    return FileResponse(out_path, media_type="audio/wav", filename="instrumental.wav")




