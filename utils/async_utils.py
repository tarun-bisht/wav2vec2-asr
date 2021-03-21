import queue
import time
import pyaudio
import asyncio
import numpy as np
import soundfile as sf
from scipy.signal import resample

class MicrophoneStreaming:
    def __init__(self, sr=16000, buffersize=1024, channels=1, dtype="float32"):
        self._sr = sr
        self._dtype = dtype
        self._buffersize = buffersize
        self._buffer = asyncio.Queue()
        self._channels = channels
        self._closed = True
        self._loop = asyncio.get_running_loop()

    async def __aenter__(self):
        await self.__open()
        return self
    
    async def __aexit__(self, type, value, traceback):
        await self.__close()
    
    def __fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._loop.call_soon_threadsafe(self._buffer.put_nowait, in_data)
        return None, pyaudio.paContinue

    async def __open(self):
        self._pyaudio_obj = pyaudio.PyAudio()
        self._stream = self._pyaudio_obj.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self._sr,
            input=True,
            frames_per_buffer=self._buffersize,
            stream_callback=self.__fill_buffer
        )
        self._closed = False

    async def __close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._closed = True
        self._pyaudio_obj.terminate()
        del(self._buffer)

    async def record_to_file(self, filename, duration=None):
        with sf.SoundFile(filename, mode='x', samplerate=self._sr, channels=self._channels) as f:
            t = time.time()
            rec = duration if duration is not None else 10
            async for block in self.generator():
                f.write(block)
                rec = duration+0 if duration is not None else duration+1
                if(time.time() - t) > rec:
                    break 

    async def generator(self):
        while not self._closed:
            try:
                chunk = await self._buffer.get()
                yield np.fromstring(chunk, dtype=self._dtype)
            except asyncio.QueueEmpty:
                self._loop.stop()
                break

class AudioStreaming:
    def __init__(self, audio_path, blocksize, sr=16000, overlap=0, padding=None, dtype="float32"):
        self._sr = sr
        self._orig_sr = sf.info(audio_path).samplerate
        self._sf_blocks = sf.blocks(audio_path,
                        blocksize=blocksize, 
                        overlap=overlap,
                        fill_value=padding,
                        dtype=dtype)

    def generator(self):
        for block in self._sf_blocks:
            chunk = self.__resample_file(block, self._orig_sr, self._sr)
            yield chunk

    def __resample_file(self, array, original_sr, target_sr):
        sample = resample(array, num=int(len(array)*target_sr/original_sr))
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