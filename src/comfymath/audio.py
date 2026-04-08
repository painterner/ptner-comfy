from typing import Any, Mapping

import os
import hashlib

import torch

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np

import folder_paths
import node_helpers

import av
import base64
from io import BytesIO
from comfy_api.latest import ComfyExtension, IO, UI

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def load(filepath: str) -> tuple[torch.Tensor, int]:
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()

            frames.append(buf)
            length += buf.shape[1]

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        return wav, sr

class LoadAudio():
    # @classmethod
    # def define_schema(cls):
    #     input_dir = folder_paths.get_input_directory()
    #     files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
    #     return IO.Schema(
    #         node_id="PtLoadAudio",
    #         search_aliases=["import audio", "open audio", "audio file"],
    #         display_name="Pt Load Audio",
    #         category="audio",
    #         essentials_category="Audio",
    #         inputs=[
    #             # IO.Combo.Input("audio", upload=IO.UploadType.audio, options=sorted(files)),
    #             IO.String.Input("audio", default=""),
    #         ],
    #         outputs=[IO.Audio.Output()],
    #     )
        
    RETURN_TYPES = ("AUDIO",)
    
    FUNCTION = "execute"
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"audio": (sorted(files), {"audio_upload": True})},
                }

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return IO.NodeOutput(audio)

    @classmethod
    def fingerprint_inputs(cls, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True

    # load = execute  # TODO: remove
    
    
NODE_CLASS_MAPPINGS = {
    "PtLoadAudio": LoadAudio,
}
