import torch
import transformers
import utils.utils as utils

print("Loading Models ...")
# tokenizer = transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# model = transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
## if models are not saved then uncomment above lines
tokenizer = torch.load("models/wave2vec2-base-960h-tokenizer")
model = torch.load("models/wave2vec2-base-960h")
print("Models Loaded ...")

def transcribe_input(tokenizer, model, inputs):
    inputs = tokenizer(inputs, return_tensors='pt').input_values
    logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim =-1)
    return tokenizer.decode(predicted_ids[0])

print("Start Transcribing...")
with utils.MicrophoneStreaming(buffersize=16000*2) as stream:
    with open("transcription.txt", "w") as f:
        for block in stream.generator():
            transcriptions = transcribe_input(tokenizer, model, block)
            if not transcriptions == "":
                f.write(transcriptions)
                print(transcriptions)
