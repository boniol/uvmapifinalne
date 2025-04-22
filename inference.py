import onnxruntime
import numpy as np
import librosa
import soundfile as sf

def separate_vocals(input_path, output_path):
    session = onnxruntime.InferenceSession("models/UVR-MDX-NET-Inst_HQ_3.onnx")
    audio, sr = librosa.load(input_path, sr=44100, mono=True)
    audio = np.expand_dims(audio.astype(np.float32), axis=0)
    result = session.run(None, {session.get_inputs()[0].name: audio})
    sf.write(output_path, result[0][0], sr)
