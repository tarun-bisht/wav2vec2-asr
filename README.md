# Wave2Net2 Sound Recognition

This repository uses wave2net2 model from hugging face transformers to create an ASR system.

## Installation

### Installing via pip
- Download and Install python (recommend 3.8)
- Create a virtual environment using `python -m venv env_name`
- enable created environment `env_path\Scripts\activate`
- Install PyTorch `pip install torch==1.8.0+cu102 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install required dependencies `pip install -r requirements.txt`

### Installing via conda
- Download and install miniconda
- Create a new virutal environment using `conda create --name env_name python==3.8`
- enable create environment `conda activate env_name`
- Install PyTorch `conda install pytorch torchaudio cudatoolkit=11.1 -c pytorch`
- install required dependencies `pip install -r requirements.txt`

## Inferencing
### via recorded audio
- run  `python asr_inference_recording.py` or `python asr_inference_recording_async.py` with parameters:
    - `--recording` or `-rec` : path to audio recording
    - `--model` or `-m`: path to saved wavenetctc model if not passed it will be downloaded (default = "")
    - `--tokenizer` or `-t` : path to saved wavenettokenizer model if not passed then it will be downloaded (default = "")
    - `--blocksize` or `-bs` : size of each audio block to be passed to model (default = 16000)
    - `--overlap` or `-ov` : overlapping between each loaded block (default = 0)
    - `--output` or `-out` : path to output file to save transcriptions. (not required)
    - `--device` or `-d` : device to use for inferencing (choices=["cpu", "cuda"] and default = cpu ie.. inference will be done in CPU) 
- example
    - `python asr_inference_recording.py --recording input/rec.ogg -bs 16000 -out output/transcription.txt`
    - `python asr_inference_recording.py --recording input/rec.ogg -bs 16000 -ov 1600 -out output/transcription.txt`
    - `python asr_inference_recording.py --recording input/rec.ogg -bs 16000 -ov 1600 -out output/transcription.txt --device gpu`
    - `python asr_inference_recording_async.py --recording input/rec.ogg -bs 16000 -ov 1600 -out output/transcription.txt --device cpu`

### via live recording
- run  `python asr_inference_live.py` or `python asr_inference_live_async.py` with parameters:
    - `--model` or `-m`: path to saved wavenetctc model if not passed it will be downloaded (default = "")
    - `--tokenizer` or `-t` : path to saved wavenettokenizer model if not passed then it will be downloaded (default = "")
    - `--blocksize` or `-bs` : size of each audio block to be passed to model (default = 16000)
    - `--output` or `-out` : path to output file to save transcriptions. (not required)
    - `--device` or `-d` : device to use for inferencing (choices=["cpu", "cuda"] and default = cpu ie.. inference will be done in CPU) 
- example
    - `python asr_inference_live.py -bs 16000 -out output/transcription.txt`
    - `python asr_inference_live.py`
    - `python asr_inference_live.py --device cuda`
    - `python asr_inference_live_async.py --device cpu`

### Comparisions
- ### GPU inference vs CPU inference
For 4min 10sec recorder audio total time taken
1. GPU (Nvidia GeForce 940MX) : 18.29sec
2. CPU : 116.85sec
- ### Async vs Non Async version
For 4min 10sec recorded audio average inference time
- With GPU (Nvidia GeForce 940MX)
    1. Async version: 0.056sec
    2. Non Async version: 0.11sec
- With CPU
    1. Async version: 0.31sec
    2. Non Async version: 0.54sec

## To do list
- Environment Setup ✔
- Inferencing with CPU ✔
- Inferencing with GPU ✔
- Asyncio Compatible ✔
- Converting model to tensorflow with onnx for inference using tensorflow
- Training and Finetuning

## Tested Platforms
- native windows 10 ✔
- windows-10 wsl2 cpu ✔
- windows-10 wsl2 gpu ✔
- Ubuntu
