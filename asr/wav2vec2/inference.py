import torch
import asyncio
import functools
import transformers
import numpy as np
from asr.wav2vec2.decoder.ctc_decoder import CTCDecoder


class Wav2Vec2ASR:
    """
    Wav2Vec2 class wrapper for speech recognition
    """

    def __init__(self, device: str = "cpu", processor_path: str = None,
                 model_path: str = None,
                 pretrained_model_name: str = "facebook/wav2vec2-base-960h",
                 beam_width: int = 5, lm_path: str = None):
        """Wave2Vec2 class constructor

        Args:
            device (str, optional): device to load model and inputs choices are 'cpu' and 'cuda. Defaults to "cpu".
            processor_path (str, optional): path to saved local processor files. Defaults to None.
            model_path (str, optional): path to saved local model. Defaults to None
            pretrained_model_name (str, optional): pretrained model name as per hugging face pretrained models to load. Defaults to "facebook/wav2vec2-base-960h".
            beam_width (int, optional): width of beam search more the number better the results but increase computation. Defaults to 5.
            lm_path (str, optional): path to saved language model. Defaults to None.
        """
        self.device = torch.device(device)
        self.processor_path = processor_path
        self.model_path = model_path
        self.pretrained_model_name = pretrained_model_name
        self.decoder = CTCDecoder(blank_idx=0,
                                  beam_width=beam_width,
                                  lm_path=lm_path)

    def load(self):
        """load models and processors
        """
        processor = (transformers.Wav2Vec2Processor.from_pretrained(self.pretrained_model_name)
                     if self.processor_path is None else torch.load(self.processor_path))
        model = (transformers.Wav2Vec2ForCTC.from_pretrained(self.pretrained_model_name)
                 if self.model_path is None else torch.load(self.model_path))
        model.eval()
        model.to(self.device)
        self.model = model
        self.processor = processor

    def _transcribe(self, inputs: torch.tensor)->str:
        """transcribe input speech and return resulting transcription

        Args:
            inputs (torch.tensor): single raw speech torch tensor (timestep,1)

        Returns:
            str: transcription of raw speech signal
        """               
        inputs = self.processor(
            inputs, return_tensors='pt').input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).logits
        outs = self.decoder(logits)
        return self.processor.decode(outs)

    async def capture_and_transcribe(self,
                                     stream_obj,
                                     started_future: asyncio.Future = None,
                                     loop=None):
        """capture streaming audio and transcribe

        Args:
            stream_obj (asr.utils.MicrophoneStreaming or asr.utils.AudioStreaming): streaming object with generator that yields audio blocks
            started_future (asyncio.Future, optional): asyncio future. Defaults to None.
            loop (optional): asyncio event loop which we can get using asyncio.get_running_loop(). Defaults to None.

        Yields:
            [generator object]: returns generator that yield outputs from streaming audio
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        async for block, status in stream_obj.generator(started_future):
            process_func = functools.partial(self._transcribe, inputs=block)
            transcriptions = await loop.run_in_executor(None, process_func)
            yield transcriptions

    async def transcribe(self, inputs:torch.tensor, loop=None):
        """transcribe and audio signal use for offline audio transcription

        Args:
            inputs (torch.tensor): raw speech signal as pytorch tensor (timestep,1)
            loop (optional): asyncio event loop which we can get using asyncio.get_running_loop(). Defaults to None.

        Returns:
            [corountine object]: coroutine object which we get await and get results asynchronously
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        process_func = functools.partial(self._transcribe, inputs=inputs)
        return await loop.run_in_executor(None, process_func)
