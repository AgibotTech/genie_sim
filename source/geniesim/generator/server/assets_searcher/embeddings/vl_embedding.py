# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import logging

from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, List

import os
import torch
import json
import time
import threading

logger = logging.getLogger(__name__)

from geniesim.generator.server.assets_searcher.models.qwen3_vl_embedding import Qwen3VLEmbedder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def tensor_to_numpy(tensor):
    """Convert tensor to numpy, handling bfloat16 type"""
    if torch.is_tensor(tensor):
        # Move to CPU first
        tensor = tensor.cpu()
        # Convert bfloat16 to float32 for numpy compatibility
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()
    return tensor


class QwenVLEmbeddings(BaseModel, Embeddings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_path: str = os.path.join(CURRENT_DIR, "../models/Qwen3-VL-Embedding-2B")
    batch_size: int = 10
    default_mode: str = "video"
    embedder: Qwen3VLEmbedder = None
    permanent_model_in_gpu: bool = False
    _last_used_time: float = 0.0
    _cleanup_thread: threading.Thread = None
    _lock: threading.Lock = None
    _shutdown: bool = False
    _idle_clear_time: float = 5.0

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        self._lock = threading.Lock()
        if self.permanent_model_in_gpu:
            self._create_embedder()
        else:
            # Start cleanup thread for non-permanent mode
            self._start_cleanup_thread()
        return self

    def _create_embedder(self):
        """Create embedder instance"""
        if self.embedder is None:
            self.embedder = Qwen3VLEmbedder(
                model_name_or_path=self.model_path,
                fps=6,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                # default_instruction="Extract embedding for the foreground object only. This video contains the same object captured from multiple viewing angles. Ignore all background elements (white/black backgrounds). Encode only the object's intrinsic, viewpoint-invariant visual features: shape, texture, material properties, structural details and usage.",
            )
            logger.info("Embedder created")

    def _clear_embedder(self):
        """Clear embedder to free GPU memory"""
        if self.embedder is not None:
            try:
                # Clear model from GPU
                if hasattr(self.embedder, "model") and self.embedder.model is not None:
                    self.embedder.model = None
                if hasattr(self.embedder, "processor") and self.embedder.processor is not None:
                    self.embedder.processor = None
                self.embedder = None
                # Force garbage collection and clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Embedder cleared from GPU memory")
            except Exception as e:
                logger.warning(f"Error clearing embedder: {e}")
                self.embedder = None

    def _start_cleanup_thread(self):
        """Start background thread to cleanup embedder after idle time"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._shutdown = False
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            logger.info("Cleanup thread started")

    def _cleanup_worker(self):
        """Background worker to cleanup embedder after 5 seconds of inactivity"""
        while not self._shutdown:
            time.sleep(1)  # Check every second
            if self.permanent_model_in_gpu:
                continue
            with self._lock:
                current_time = time.time()
                if self.embedder is not None and self._last_used_time > 0:
                    idle_time = current_time - self._last_used_time
                    if idle_time >= self._idle_clear_time:
                        logger.info(f"Embedder idle for {idle_time:.2f}s, clearing from GPU")
                        self._clear_embedder()

    def _ensure_embedder(self):
        """Ensure embedder exists, create if needed"""
        if self.embedder is None:
            self._create_embedder()
        self._last_used_time = time.time()

    def embed_documents(self, texts: List[str | dict]) -> List[List[float]]:
        documents_num = {
            "text": 0,
            "image": 0,
            "video": 0,
        }
        modified_texts = []
        for i, text in enumerate(texts):
            try:
                text = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                pass
            if isinstance(text, str):
                documents_num["text"] += 1
                instruction = "Extract embedding for the text. Encode only the text described object's intrinsic: shape, texture, material properties, structural details and usage."
                modified_texts.append({"text": text["text"], "instruction": instruction})
            elif isinstance(text, dict):
                if "text" in text.keys():
                    documents_num["text"] += 1
                    instruction = "Extract embedding for the text. Encode only the text described object's intrinsic: shape, texture, material properties, structural details and usage."
                    modified_texts.append({"text": text["text"], "instruction": instruction})
                elif "image" in text.keys():
                    documents_num["image"] += 1
                    semantic_name = text.get("semantic_name", "")
                    semantic_description = text.get("semantic_description", "")
                    instruction = "Extract embedding for the foreground object only. Ignore all background elements (white/black backgrounds). Encode only the object's intrinsic, viewpoint-invariant visual features: shape, texture, material properties, structural details and usage. The provided text's semantic name and semantic description are the description about the object generated by other VLM, BUT THE VISION IS FIRST PRIORITY."
                    text_element = text.get("text", "")
                    if semantic_name:
                        text_element += f"semantic_name: {semantic_name};"
                    if semantic_description:
                        text_element += f"semantic_description: {semantic_description};"
                    modified_texts.append({"image": text["image"], "instruction": instruction, "text": text_element})
                elif "video" in text.keys():
                    documents_num["video"] += 1
                    semantic_name = text.get("semantic_name", "")
                    semantic_description = text.get("semantic_description", "")
                    instruction = "Extract embedding for the foreground object only. Ignore all background elements (white/black backgrounds). Encode only the object's intrinsic, viewpoint-invariant visual features: shape, texture, material properties, structural details and usage. The provided text's semantic name and semantic description are the description about the object generated by other VLM, BUT THE VISION IS FIRST PRIORITY."
                    text_element = text.get("text", "")
                    if semantic_name:
                        text_element += f"semantic_name: {semantic_name};"
                    if semantic_description:
                        text_element += f"semantic_description: {semantic_description};"
                    modified_texts.append(
                        {"video": text["video"], "fps": 6, "instruction": instruction, "text": text_element}
                    )
                else:
                    raise ValueError(f"Invalid text: {text}")
            else:
                raise ValueError(f"Invalid text: {text}")
        logger.info(
            f"Embedding {documents_num['text']} text documents, {documents_num['image']} image documents, {documents_num['video']} video documents in total"
        )
        all_doc_embeddings = []
        for i in range(0, len(modified_texts), self.batch_size):
            logger.info(f"Embedding batch {i} of {len(modified_texts)}")
            documents_batch = modified_texts[i : i + self.batch_size]
            # Call batch embedding API
            with self._lock:
                self._ensure_embedder()
                doc_resp = self.embedder.process(documents_batch)
                self._ensure_embedder()
            doc_resp = tensor_to_numpy(doc_resp)
            # Extract each embedding in order and append to aggregated list
            for doc_emb in doc_resp:
                all_doc_embeddings.append(doc_emb)
        return all_doc_embeddings

    def embed_query(self, query: str | dict) -> List[float]:
        if isinstance(query, str):
            if os.path.isfile(query):
                if os.path.exists(query):
                    query = {"video": query, "fps": 6}
                else:
                    raise FileNotFoundError(f"File does not exist: {query}")
            else:
                instruction = "Extract embedding for the text. Encode only the text described object's intrinsic: shape, texture, material properties, structural details and usage."
                query = {"text": query, "instruction": instruction}
        elif isinstance(query, dict):
            if "text" in query.keys():
                pass
            elif "image" in query.keys():
                pass
            elif "video" in query.keys():
                pass
            else:
                raise ValueError(f"Invalid query: {query}")
        else:
            raise ValueError(f"Invalid query: {query}")

        # Ensure embedder exists and update last used time
        with self._lock:
            self._ensure_embedder()
            result = self.embedder.process([query])[0]
        return tensor_to_numpy(result)
