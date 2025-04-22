from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
from inference import separate_vocals
from urllib.request import Request, urlopen

# Pobieranie tokenu z ENV
token = os.environ.get("HUGGINGFACE_TOKEN")

model_url = "https://huggingface.co/Kuielito/UVR/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx"
model_path = "models/UVR-MDX-NET-Inst_HQ_3.onnx"

os.makedirs("models", exist_ok=True)

if not os.path.exists(model_path):
    print("Downloading model from Hugging Face...")
    req = Request(model_url)
    req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req) as response, open(model_path, "wb") as out_file:
        out_file.write(response.read())
    print("Model downloaded.")

app = FastAPI()

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    input_path = f"input_audio/{file.filename}"
    output_path = f"output_audio/{file.filename}_inst.wav"

    os.makedirs("input_audio", exist_ok=True)
    os.makedirs("output_audio", exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    separate_vocals(input_path, output_path)

    return FileResponse(output_path, media_type="audio/wav", filename="instrumental.wav")

