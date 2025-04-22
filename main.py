from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import shutil
import urllib.request
from urllib.request import Request
from inference import separate_vocals  # zakładam, że masz tę funkcję w osobnym pliku

app = FastAPI()

# === Pobieranie modelu z Hugging Face ===

# Pobierz token z ENV
token = os.environ.get("HUGGINGFACE_TOKEN")

# Link do modelu (działające repo)
model_url = "https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx"
model_path = "models/UVR-MDX-NET-Inst_HQ_3.onnx"

# Upewnij się, że folder 'models' istnieje
os.makedirs("models", exist_ok=True)

# Pobierz model jeśli go nie ma
if not os.path.exists(model_path):
    print("Downloading model from Hugging Face...")
    req = Request(model_url)
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as response, open(model_path, "wb") as out_file:
        out_file.write(response.read())
    print("Model downloaded.")


# === Endpoint do separacji wokalu ===

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    input_dir = "input_audio"
    output_dir = "output_audio"
    input_path = f"{input_dir}/{file.filename}"
    output_path = f"{output_dir}/{file.filename}_inst.wav"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Zapisz plik
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Przetwórz audio
    separate_vocals(input_path, output_path)

    # Zwróć wynik
    return FileResponse(output_path, media_type="audio/wav", filename="instrumental.wav")



