import time
import asyncio
import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample

import torch
from torchaudio.transforms import Resample


class MicrophoneCaptureFailed(Exception):
    pass


class MicrophoneStreaming:
    def __init__(self, sr=16000, blocksize=1024, channels=1, device=None, loop=None, dtype="float32"):
        self._sr = sr
        self._channels = channels
        self._device = device
        self._buffer = asyncio.Queue()
        self._buffersize = blocksize
        self._dtype = dtype
        self._loop = loop
        
    def __callback(self, indata, frame_count, time_info, status):
        self._loop.call_soon_threadsafe(self._buffer.put_nowait, (indata.copy(), status))
    
    async def record_to_file(self, filename, duration=None):
        with sf.SoundFile(filename, mode='x', samplerate=self._sr, channels=self._channels) as f:
            t = time.time()
            rec = duration if duration is not None else 10
            async for block, status in self.generator():
                f.write(block)
                rec = duration+0 if duration is not None else duration+1
                if(time.time() - t) > rec:
                    break 

    async def generator(self, future: asyncio.Future = None):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        stream = sd.InputStream(
            samplerate=self._sr,
            device=self._device,
            channels=self._channels,
            callback=self.__callback,
            dtype=self._dtype,
            blocksize=self._buffersize)
        with stream:
            if not stream.active:
                # if it was not called start() or exception was raised
                # in the audio callback
                if future: 
                    # if the future is waiting for the start or any failure
                    # set the exception
                    future.set_exception(f"Could not open the {self._device} capture device")
                
                # coroutine also will be notified
                raise MicrophoneCaptureFailed
            else:
                if future:
                    # if the future is waiting for the start or any failure
                    # set True meaning that the microphone was successfully opened
                    future.set_result(True)
            
            while stream.active:
                indata, status = await self._buffer.get()
                yield indata.squeeze(), status


class AudioStreaming:
    def __init__(self, audio_path, blocksize, sr=16000, overlap=0, padding=None, dtype="float32"):
        assert blocksize >= 0, "blocksize cannot be 0 or negative"
        self._sr = sr
        self._orig_sr = sf.info(audio_path).samplerate
        self._sf_blocks = sf.blocks(audio_path,
                        blocksize=blocksize, 
                        overlap=overlap,
                        fill_value=padding,
                        dtype=dtype)

    async def generator(self, future: asyncio.Future=None):
        for block in self._sf_blocks:
            chunk = await self.__resample_file(block, self._orig_sr, self._sr)
            yield chunk, self._orig_sr

    async def __resample_file(self, array, original_sr, target_sr):
        resampling_transform = Resample(orig_freq=original_sr,
                                        new_freq=target_sr)

        sample = resampling_transform(torch.Tensor([array])).squeeze()
        return sample


class AudioReader:
    def __init__(self, audio_path, sr=16000, dtype="float32"):
        self._sr = sr
        self._dtype = dtype
        self._audio_path = audio_path
    
    def read(self):
        data, sr = sf.read(self._audio_path, dtype=self._dtype)
        data = self.__resample_file(data, sr, self._sr)
        return data, sr

    def __resample_file(self, array, original_sr, target_sr):
        return resample(array, num=int(len(array)*target_sr/original_sr))