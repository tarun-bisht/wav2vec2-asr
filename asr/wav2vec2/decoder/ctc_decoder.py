import torch
import numpy as np
from asr.language_model import LanguageModel
from asr.utils import BeamSearchDecoder
from asr.wave2vec2.vocab import vocab_list


class CTCDecoder:
    def __init__(self, blank_idx: int = 0, beam_width: int = 100, lm_path: str = None):
        """constructor

        Args:
            blank_idx (int, optional): index of ctc blank token. Defaults to 0.
            beam_width (int, optional): beam width to search larget the value gives more accurate decoding costing computation. Defaults to 100.
            lm_path (str, optional): path to langugage model folder with unigram and bigrams. Defaults to None.
        """        
        lm = None
        if beam_width <= 1:
            self.mode = "greedy"
        else:
            self.mode = "beam"
            if lm_path is not None:
                self.mode = "beam_lm"
                lm = LanguageModel()
                lm.load(lm_path)
        self._beam_search = BeamSearchDecoder(vocab_list[1:],
                                              blank_idx,
                                              beam_width,
                                              lm=lm)

    def __call__(self, logits: torch.tensor):
        return self.decode(logits)

    def decode(self, logits: torch.tensor):
        """decode logits using greedy method or beam search if beam width <= 1 then greedy else beam search.

        Args:
            logits (torch.tensor): logits from model outputs

        Returns:
            np.array: ctc decoded output
        """
        out_proba = torch.nn.functional.softmax(logits, dim=-1)[0]
        if self.mode == "greedy":
            out = self._greedy_path(out_proba)
        elif self.mode == "beam":
            out = self._beam_search(out_proba.numpy())[0]
        elif self.mode == "beam_lm":
            out = self._beam_search(out_proba.numpy())[0]
        else:
            out = None
            raise ValueError(
                "Mode not defined mode choices [greedy, beam and beam_lm]")
        return out

    def _greedy_path(self, logits: torch.tensor) -> torch.tensor:
        """max decoding ctc output by taking maximum probabilities from each timestep

        Args:
            logits (torch.tensor): softmax logits from model

        Returns:
            torch.tensor: max decoded outputs
        """
        out_proba = torch.nn.functional.softmax(logits, dim=-1)
        return torch.argmax(out_proba, axis=1).numpy()
