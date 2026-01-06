# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import logging

from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, ConfigDict, model_validator
from openai import OpenAI
from typing_extensions import Self

logger = logging.getLogger(__name__)
from dashscope import TextEmbedding
import dashscope


class TextEmbeddings(BaseModel, Embeddings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: OpenAI = None
    model: str = "qwen3-embedding:8b"
    api_key: str = "ollama"
    dimension: int = 2048
    base_url: str = ""
    batch_size: int = 10

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.dimension not in [512, 1024, 2048, 4096]:
            raise ValueError(f"Invalid dimension: {self.dimension}. Must be one of [512, 1024, 2048, 4096]")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input text list.
        Args:
            texts (List[str]): List of texts to generate embeddings for.
        Returns:
            List[List[float]]: List of embeddings for each document in the input list. Each embedding is represented as a list of float values.
        """
        all_doc_embeddings = []
        batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            documents_batch = texts[i : i + batch_size]
            doc_resp = self.client.embeddings.create(
                model=self.model,
                input=documents_batch,
                # Set vector dimension
                dimensions=self.dimension,
            )
            # Extract each embedding in order and append to aggregated list
            for doc_emb in doc_resp.data:
                all_doc_embeddings.append(doc_emb.embedding)
        return all_doc_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for input text.
        Args:
            query (str): Text to generate embedding for.
        Return:
            embeddings (List[float]): Embedding of the input text, a list of float values.
        """
        embeddings = self.client.embeddings.create(model=self.model, input=query, dimensions=self.dimension)
        return embeddings.data[0].embedding


class DashscopeTextEmbeddings(BaseModel, Embeddings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str = "text-embedding-v4"
    api_key: str = ""
    dimension: int = 2048
    batch_size: int = 10

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.dimension not in [512, 1024, 2048]:
            raise ValueError(f"Invalid dimension: {self.dimension}. Must be one of [512, 1024, 2048]")
        if self.api_key == "":
            raise ValueError("API key not set")
        dashscope.api_key = self.api_key
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input text list.
        Args:
            texts (List[str]): List of texts to generate embeddings for.
        Returns:
            List[List[float]]: List of embeddings for each document in the input list. Each embedding is represented as a list of float values.
        """
        all_doc_embeddings = []
        all_token = 0
        for i in range(0, len(texts), self.batch_size):
            documents_batch = texts[i : i + self.batch_size]
            # Call batch embedding API
            doc_resp = TextEmbedding.call(model=self.model, input=documents_batch, dimension=self.dimension)
            all_token += doc_resp.usage["total_tokens"]
            # Extract each embedding in order and append to aggregated list
            for doc_emb in doc_resp.output["embeddings"]:
                all_doc_embeddings.append(doc_emb["embedding"])
        return all_doc_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for input text.
        Args:
            query (str): Text to generate embedding for.
        Return:
            embeddings (List[float]): Embedding of the input text, a list of float values.
        """
        query_resp = TextEmbedding.call(model=self.model, input=query, dimension=self.dimension)
        query_embedding = query_resp.output["embeddings"][0]["embedding"]
        return query_embedding
