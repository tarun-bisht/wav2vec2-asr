import argparse
import asyncio
import functools
from asr.utils import MicrophoneStreaming
from asr.wav2vec2 import Wav2Vec2ASR

parser = argparse.ArgumentParser(description="ASR with live audio")
parser.add_argument("--model", "-m", default=None, required=False,
                    help="Trained Model local path")
parser.add_argument("--processor", "-t", default=None, required=False,
                    help="Local asr processor path")
parser.add_argument("--blocksize", "-bs", default=16000, type=int, required=False,
                    help="Size of each audio block to be passed to model")
parser.add_argument("--output", "-out", required=False,
                    help="path to save resultant transcriptions")
parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                    help="device to use for inferencing")
parser.add_argument("--beam_width", "-bw", default=1, type=int, required=False,
                    help="beam width to use for beam search decoder during inferencing")
parser.add_argument("--lm", "-l", default=None, required=False,
                    help="Trained lm folder path with unigram and bigram files")
parser.add_argument("--pretrained_wavenet_model_name", "-pwmn", default="facebook/wav2vec2-base-960h",
                    type=str, required=False, help="Pretrained wavenet model name")

args = parser.parse_args()

asr = Wav2Vec2ASR(device=args.device,
                  processor_path=args.processor,
                  model_path=args.model,
                  pretrained_wavenet_model_name=args.pretrained_wavenet_model_name
                  beam_width=args.beam_width,
                  lm_path=args.lm)

print("Loading Models ...")
asr.load()
print("Models Loaded ...")


def write_to_file(output_file, transcriptions):
    output_file.write(transcriptions)


def print_transcription(transcription):
    print(transcription, end=" ")
    sys.stdout.flush()


async def main(output_file=None):
    loop = asyncio.get_running_loop()
    stream = MicrophoneStreaming(blocksize=args.blocksize, loop=loop)
    async for transcription in asr.capture_and_transcribe(stream, loop=loop):
        if not transcription == "":
            print_func = functools.partial(
                print_transcription, transcription=transcription)
            await loop.run_in_executor(None, print_func)
            if output_file is not None:
                write_func = functools.partial(write_to_file, output_file=output_file,
                                               transcriptions=transcriptions)
                await loop.run_in_executor(None, write_func)

if __name__ == "__main__":
    print("Start Transcribing...")
    try:
        if args.output:
            with open(args.output, "w") as f:
                asyncio.run(main(f))
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited")
