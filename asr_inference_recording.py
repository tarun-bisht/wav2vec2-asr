import torch
import transformers
import utils.utils as utils
import argparse
import time

parser = argparse.ArgumentParser(description="ASR with recorded audio")
parser.add_argument("--recording", "-rec", required=True,
                    help="Trained Model path")
parser.add_argument("--model", "-m", default="",required=False,
                    help="Trained Model path")
parser.add_argument("--tokenizer", "-t", default="", required=False,
                    help="Trained tokenizer path")
parser.add_argument("--blocksize", "-bs", default=16000, type=int, required=False,
                    help="Size of each audio block to be passed to model")
parser.add_argument("--overlap", "-ov", default=0, type=int, required=False,
                    help="Overlap between blocks")
parser.add_argument("--output", "-out", required=False,
                    help="Output Path for saving resultant transcriptions")
parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                    help="device to use for inferencing")

args = parser.parse_args()

device = torch.device(args.device)

print("Loading Models ...")
tokenizer = (transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h") 
                if args.tokenizer == "" else torch.load(args.tokenizer))
model = (transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") 
            if args.model == "" else torch.load(args.model))

model.eval()
model.to(device)
print("Models Loaded ...")

def transcribe_input(tokenizer, model, inputs):
    inputs = tokenizer(inputs, return_tensors='pt').input_values.to(device)
    logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim =-1)
    return tokenizer.decode(predicted_ids[0])

print("Start Transcribing...")
stream = utils.AudioStreaming(audio_path=args.recording, 
                            blocksize=args.blocksize, 
                            overlap=args.overlap, 
                            padding=0, 
                            sr=16000, 
                            dtype="float32")

if args.output:
    with open(args.output, "w") as f:
        start = time.time()
        for block in stream.generator():
            transcriptions = transcribe_input(tokenizer, model, block)
            if not transcriptions == "":
                f.write(transcriptions)
                print(transcriptions)
        end = time.time()
        print(f"Time Taken: {end-start}sec")
else:
    start = time.time()
    for block in stream.generator():
        transcriptions = transcribe_input(tokenizer, model, block)
        if not transcriptions == "":
            print(transcriptions)
    end = time.time()
    print(f"Time Taken: {end-start}sec")

