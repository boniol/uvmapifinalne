from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os, shutil
from inference import separate_vocals

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
