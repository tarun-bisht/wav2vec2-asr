import argparse
import asyncio
import functools
from asr.utils import AudioReader
from asr.wav2vec2 import Wav2Vec2ASR

parser = argparse.ArgumentParser(
    description="ASR with recorded audio (offline)")
parser.add_argument("--recording", "-rec", required=True,
                    help="path to recording file")
parser.add_argument("--model", "-m", default=None, required=False,
                    help="path to local saved model")
parser.add_argument("--processor", "-t", default=None, required=False,
                    help="path to local saved processor")
parser.add_argument("--output", "-out", required=False,
                    help="path to save resultant transcriptions")
parser.add_argument("--lm", "-l", default=None, required=False,
                    help="Trained lm folder path with unigram and bigram files")
parser.add_argument("--device", "-d", default='cpu', nargs='?', choices=['cuda', 'cpu'], required=False,
                    help="device to use for inferencing")
parser.add_argument("--beam_width", "-bw", default=1, type=int, required=False,
                    help="beam width to use for beam search decoder during inferencing")
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


async def main():
    loop = asyncio.get_running_loop()
    reader = AudioReader(audio_path=args.recording,
                         sr=16000,
                         dtype="float32")
    inputs, sr = reader.read()
    transcriptions = await asr.transcribe(inputs, loop=loop)
    print(transcriptions)
    if args.output:
        with open(args.output, "w") as f:
            f.write(transcriptions)


if __name__ == "__main__":
    print("Start Transcribing...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited")
