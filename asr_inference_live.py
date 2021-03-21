import torch
import transformers
import utils.utils as utils
import argparse

parser = argparse.ArgumentParser(description="ASR with live audio")
parser.add_argument("--model", "-m", default="",required=False,
                    help="Trained Model path")
parser.add_argument("--tokenizer", "-t", default="", required=False,
                    help="Trained tokenizer path")
parser.add_argument("--blocksize", "-bs", default=16000, type=int, required=False,
                    help="Size of each audio block to be passed to model")
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
    with utils.MicrophoneStreaming(buffersize=args.blocksize) as stream:
        for block in stream.generator():
            transcriptions = transcribe_input(tokenizer=tokenizer, 
                                            model=model, 
                                            inputs=block)
            if not transcriptions == "":
                print_transcriptions(transcriptions=transcriptions)
                if output_file is not None:
                    write_to_file(output_file=output_file, 
                                    transcriptions=transcriptions)

if __name__=="__main__":
    print("Start Transcribing...")
    try:
        if args.output:
            with open(args.output, "w") as f:
                capture_and_transcribe(f)
        else:
            capture_and_transcribe()
    except KeyboardInterrupt:
        print("Exited")