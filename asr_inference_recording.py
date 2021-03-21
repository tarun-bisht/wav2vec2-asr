import torch
import transformers
import utils.utils as utils
import argparse
import time
import numpy as np

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

def print_transcriptions(transcriptions):
    print(transcriptions, end=" ")

def write_to_file(output_file, transcriptions):
    output_file.write(transcriptions)

def capture_and_transcribe(output_file=None):
    infer_time = []
    for block in stream.generator():
        start = time.time()
        transcriptions = transcribe_input(tokenizer=tokenizer, 
                                        model=model, 
                                        inputs=block)
        end = time.time()
        infer_time.append(end-start)
        if not transcriptions == "":
            print_transcriptions(transcriptions=transcriptions)
            if output_file is not None:
                write_to_file(output_file=output_file, 
                            transcriptions=transcriptions)
    return np.mean(infer_time)


if __name__=="__main__":
    stream = utils.AudioStreaming(audio_path=args.recording, 
                                blocksize=args.blocksize, 
                                overlap=args.overlap, 
                                padding=0, 
                                sr=16000, 
                                dtype="float32")

    print("Start Transcribing...")
    try:
        start = time.time()
        if args.output:
            with open(args.output, "w") as f:
                infer_time = capture_and_transcribe(f)
        else:
            infer_time = capture_and_transcribe()
        end = time.time()
        print(f"Total Time Taken: {end-start}sec")
        print(f"Average Inference Time: {infer_time}sec")
    except KeyboardInterrupt:
        print("Exited")

