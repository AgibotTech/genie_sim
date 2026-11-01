# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Qwen3-VL embedding model implementation vendored for local asset search.

Based on QwenLM/Qwen3-VL-Embedding's qwen3_vl_embedding.py. Model weights remain
separate and are selected with --model-path or QWEN3_VL_MODEL.
"""

import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging

from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.utils.generic import check_model_inputs
from qwen_vl_utils.vision_process import process_vision_info

logger = logging.getLogger(__name__)

# Constants for configuration
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = "<|endoftext|>"

# Define output structure for embeddings
@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None

# Define model class to compute embeddings
class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Extract video features from model
    def get_video_features(self, pixel_values_videos: torch.FloatTensor,
                           video_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    # Extract image features from model
    def get_image_features(self, pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules accessible through properties
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # Forward pass through model with input parameters
    # @check_model_inputs
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        # Pass inputs through the model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        # Return the model output
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

def sample_frames(frames: List[Union[str, Image.Image]], num_segments: int, max_segments: int) -> List[str]:
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    # Create a list of sampled frames
    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            sampled_frames.append(frames[frame_idx])
        except:
            break
    # Ensure the sampled list meets the required segment count
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]

# Define embedder class for processing inputs and generating embeddings
class Qwen3VLEmbedder():
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames

        self.default_instruction = default_instruction

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right'
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'attention_mask': inputs.get('attention_mask')
        }

    # Truncate token sequence to a specified max length
    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids

        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for token_idx in token_ids if token_idx in special_token_ids)
        num_non_special_to_keep = max_length - num_special

        final_token_ids = []
        non_special_kept_count = 0
        # Ensure retention of special tokens while truncating the rest
        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
        return final_token_ids

    # Format input based on provided text, image, video, and instruction
    def format_model_input(
        self, text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        # Initialize conversation with system prompts
        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]

        # Add text, image, or video content to conversation
        if not text and not image and not video:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        if video:
            video_content = None
            video_kwargs = { 'total_pixels': self.total_pixels }
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(video, str):
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames,}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            # Add video input details to content
            if video_content:
                content.append({
                    'type': 'video', 'video': video_content,
                    **video_kwargs
                })

        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")

            # Add image input details to content
            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        if text:
            content.append({'type': 'text', 'text': text})

        return conversation

    # Preprocess input conversations for model consumption
    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}],
                add_generation_prompt=True, tokenize=False
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=video_metadata, truncation=True,
            max_length=self.max_length, padding=True, do_resize=False, return_tensors='pt',
            **video_kwargs
        )
        return inputs

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> tuple:
        conversations = [self.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            video=ele.get('video'),
            instruction=ele.get('instruction'),
            fps=ele.get('fps'),
            max_frames=ele.get('max_frames')
        ) for ele in inputs]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}

        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

        # Normalize the embeddings if specified
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
