import queue
import time
import pyaudio
import numpy as np
import soundfile as sf
from scipy.signal import resample

class MicrophoneStreaming:
    def __init__(self, sr=16000, buffersize=1024, channels=1, dtype="float32"):
        self._sr = sr
        self._dtype = dtype
        self._buffersize = buffersize
        self._buffer = queue.Queue()
        self._channels = channels
        self._closed = True

    def __enter__(self):
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
        return self
    
    def __exit__(self, type, value, traceback):
        self._stream.stop_stream()
        self._stream.close()
        self._closed = True
        self._buffer.put(None)
        self._pyaudio_obj.terminate()
    
    def __fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buffer.put(in_data)
        return None, pyaudio.paContinue

    def record_to_file(self, filename, duration=None):
        with sf.SoundFile(filename, mode='x', samplerate=self._sr, channels=self._channels) as f:
            t = time.time()
            rec = duration if duration is not None else 10
            for block in self.generator():
                f.write(block)
                rec = duration+0 if duration is not None else duration+1
                if(time.time() - t) > rec:
                    break 

    def generator(self):
        while not self._closed:
            chunk = self._buffer.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buffer.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            byte_data = b"".join(data)
            yield np.fromstring(byte_data, dtype=self._dtype)

class AudioStreaming:
    def __init__(self, audio_path, blocksize, sr=16000, overlap=0, padding=None, dtype="float32"):
        self._sr = sr
        self._dtype = dtype
        self._audio_path = audio_path
        self._blocksize = blocksize
        self._overlap = overlap
        self._padding = padding

    def generator(self):
        sr = sf.info(self._audio_path).samplerate
        sf_blocks = sf.blocks(self._audio_path, 
                        blocksize=self._blocksize, 
                        overlap=self._overlap,
                        fill_value=self._padding, 
                        dtype=self._dtype)
        for block in sf_blocks:
            yield self.__resample_file(block, sr, self._sr)

    def __resample_file(self, array, original_sr, target_sr):
        return resample(array, num=int(len(array)*target_sr/original_sr))


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