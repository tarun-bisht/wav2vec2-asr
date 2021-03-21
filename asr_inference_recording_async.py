import torch
import transformers
import utils.async_utils as utils
import argparse
import time
import asyncio
import functools
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

stream = utils.AudioStreaming(audio_path=args.recording, 
                            blocksize=args.blocksize, 
                            overlap=args.overlap, 
                            padding=0, 
                            sr=16000, 
                            dtype="float32")


def transcribe_input(tokenizer, model, inputs):
    inputs = tokenizer(inputs, return_tensors='pt').input_values.to(device)
    logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim =-1)
    return tokenizer.decode(predicted_ids[0])

def print_transcriptions(transcriptions):
    print(transcriptions, end=" ")

def write_to_file(output_file, transcriptions):
    output_file.write(transcriptions)

async def capture_and_transcribe(output_file=None):
    infer_time = []
    loop = asyncio.get_running_loop()
    for block in stream.generator():
        start = time.time()
        process_func = functools.partial(transcribe_input, 
                                tokenizer=tokenizer, 
                                model=model, 
                                inputs=block)
        transcriptions = await loop.run_in_executor(None, process_func)
        end = time.time()
        infer_time.append(end-start)
        if not transcriptions == "":
            print_func = functools.partial(print_transcriptions, transcriptions=transcriptions)
            start = time.time()
            await loop.run_in_executor(None, print_func)
            end = time.time()
            infer_time.append(end-start)
            if output_file is not None:
                write_func = functools.partial(write_to_file, output_file=output_file, 
                                                transcriptions=transcriptions)
                await loop.run_in_executor(None, write_func)
                
    return np.mean(infer_time)

if __name__=="__main__":
    print("Start Transcribing...")
    try:
        start = time.time()
        if args.output:
            with open(args.output, "w") as f:
                infer_time = asyncio.run(capture_and_transcribe(f))
        else:
            infer_time = asyncio.run(capture_and_transcribe())
        end = time.time()
        print(f"Total Time Taken: {end-start}sec")
        print(f"Average Inference Time: {infer_time}sec")
    except KeyboardInterrupt:
        print("Exited")