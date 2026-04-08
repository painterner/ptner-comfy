from typing import Any, Mapping

import os
import hashlib

import torch

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np

import folder_paths
import node_helpers
import logging

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
    
    
class AudioConcat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioConcat",
            search_aliases=["join audio", "combine audio", "append audio"],
            display_name="Audio Concat",
            description="Concatenates the audio1 to audio2 in the specified direction.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio1"),
                IO.Audio.Input("audio2"),
                IO.Combo.Input(
                    "direction",
                    options=['after', 'before'],
                    default="after",
                    tooltip="Whether to append audio2 after or before audio1.",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio1, audio2, direction) -> IO.NodeOutput:
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]

        if waveform_1.shape[1] == 1:
            waveform_1 = waveform_1.repeat(1, 2, 1)
            logging.info("AudioConcat: Converted mono audio1 to stereo by duplicating the channel.")
        if waveform_2.shape[1] == 1:
            waveform_2 = waveform_2.repeat(1, 2, 1)
            logging.info("AudioConcat: Converted mono audio2 to stereo by duplicating the channel.")

        waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)

        if direction == 'after':
            concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
        elif direction == 'before':
            concatenated_audio = torch.cat((waveform_2, waveform_1), dim=2)

        return IO.NodeOutput({"waveform": concatenated_audio, "sample_rate": output_sample_rate})

    concat = execute  # TODO: remove
    
    
class PtLoadAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return IO.Schema(
            node_id="PtLoadAudio",
            search_aliases=["import audio", "open audio", "audio file"],
            display_name="Pt Load Audio",
            category="ptaudio",
            essentials_category="PtAudio",
            inputs=[
                # IO.Combo.Input("audio", options=sorted(files)),
                IO.String.Input("audio", default="", tooltip="Path to the audio file relative to the input directory."),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        # logging.info(f"PtLoadAudio: Loading audio from {audio}")
        audio_path = folder_paths.get_annotated_filepath(audio)
        # logging.info(f"PtLoadAudio: Loading audio from {audio_path}")
        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return IO.NodeOutput(audio)

    @classmethod
    def fingerprint_inputs(cls, audio):
        # logging.info(f"PtLoadAudio: chaning audio from {audio}")
        image_path = folder_paths.get_annotated_filepath(audio)
        # logging.info(f"PtLoadAudio: chaning audio from {image_path}")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, audio):
        # logging.info(f"PtLoadAudio: validating audio from {audio}")
        # if not folder_paths.exists_annotated_filepath(audio):
        #     return "Invalid audio file: {}".format(audio)
        return True

    load = execute  # TODO: remove

# class LoadAudio():
#     @classmethod
#     def define_schema(cls):
#         input_dir = folder_paths.get_input_directory()
#         files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
#         return IO.Schema(
#             node_id="PtLoadAudio",
#             search_aliases=["import audio", "open audio", "audio file"],
#             display_name="Pt Load Audio",
#             category="audio",
#             essentials_category="Audio",
#             inputs=[
#                 IO.Combo.Input("audio", options=sorted(files)),
#                 # IO.String.Input("audio", default=""),
#             ],
#             outputs=[IO.Audio.Output()],
#         )
        
#     # RETURN_TYPES = ("AUDIO",)
    
#     # FUNCTION = "execute"
    
#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = folder_paths.get_input_directory()
#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
#         return {"required":
#                     {"audio": (sorted(files), {"audio_upload": True})},
#                 }

#     @classmethod
#     def execute(cls, audio) -> IO.NodeOutput:
#         audio_path = folder_paths.get_annotated_filepath(audio)
#         waveform, sample_rate = load(audio_path)
#         audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
#         return IO.NodeOutput(audio)

#     @classmethod
#     def fingerprint_inputs(cls, audio):
#         audio_path = folder_paths.get_annotated_filepath(audio)
#         m = hashlib.sha256()
#         with open(audio_path, 'rb') as f:
#             m.update(f.read())
#         return m.digest().hex()

#     @classmethod
#     def validate_inputs(cls, audio):
#         if not folder_paths.exists_annotated_filepath(audio):
#             return "Invalid audio file: {}".format(audio)
#         return True

#     load = execute  # TODO: remove
    
    
NODE_CLASS_MAPPINGS = {
    "PtLoadAudio": PtLoadAudio,
}
