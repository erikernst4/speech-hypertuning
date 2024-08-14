import random
from typing import Any, Dict, Optional, Union

import numpy as np
import soundfile as sf
import torch


def ProcessorReadAudio(
    x,
    state,
    input=None,
    output=None,
    max_length: Union[float, int] = None,
    mono: bool = True,
    sampling_method: Optional[str] = None,
):
    def read_sample(
        x,
        state: Dict[str, Any],
        max_length: Union[float, int],
        mono: bool,
        sampling_method: str,
    ):
        if max_length is not None:
            audio_info = sf.info(x[input])
            desired_frames = int(max_length * audio_info.samplerate)
            total_frames = audio_info.frames
            if total_frames > desired_frames:
                if sampling_method == "random":
                    start = random.randint(0, total_frames - desired_frames)
                elif sampling_method == "starting_chunk":
                    start = 0
                else:
                    raise ValueError("Not a valid sampling method for chunking")
                stop = start + desired_frames
            else:
                start = 0
                stop = None
            if 'chunk_idx' in state:
                # This is for ordered reading in chunks when doing evals
                start = int(state['chunk_idx'] * desired_frames)
                stop = start + desired_frames
        else:
            start = 0
            stop = None
        if 'start' in x:
            start = x['start']
        if 'stop' in x:
            stop = x['stop']
        x['start'] = start
        x['stop'] = stop
        wav, _ = sf.read(x[input], start=start, stop=stop, dtype=np.float32)
        if (wav.ndim == 2) and mono:
            wav = np.mean(wav, axis=-1)
        return wav

    if sampling_method is None:
        sampling_method = "random"

    try:
        wav = read_sample(x, state, max_length, mono, sampling_method)
    except:
        print('Failed reading {}'.format(x))
        wav = None

    x["wav"] = wav
    x["wav_lens"] = np.array([wav.shape[0]], dtype=int)
    return x, state


def ProcessorLoadUpstreamEmbedding(
    x,
    state: Dict[str, Any],
):
    embedding_path = x["embedding_filename"]
    with torch.no_grad():
        x['upstream_embedding'] = torch.load(
            embedding_path,
            weights_only=True,
        )
        x['upstream_embedding'] = x['upstream_embedding'].to("cuda")
    x["upstream_embedding_precalculated"] = True
    return x, state
