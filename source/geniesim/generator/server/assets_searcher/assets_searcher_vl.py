# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import json
import sys
import logging
from datetime import datetime
import traceback
import time
import threading
from typing import List, Dict, Union, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(SOURCE_DIR)

from geniesim.generator.server.assets_searcher.embeddings.vl_embedding import QwenVLEmbeddings
from geniesim.generator.server.assets_searcher.models.qwen3_vl_reranker import Qwen3VLReranker
import hashlib
import shutil

from geniesim.assets import (
    ASSETS_INDEX,
    ASSETS_INDEX_HASH,
    ASSETS_PATH,
)  # If this fails please check assets folder at source/geniesim/assets

from geniesim.generator.server.assets_searcher.assets_searcher import AssetVectorDB
import torch

CHROMA_DB_CACHE_PATH = os.path.join("/tmp", "chromadb_cache_vl")


# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def calculate_file_hash(file_path, algorithm="sha256", chunk_size=8192):
    """Calculate the hash of a file"""
    with open(file_path, "rb") as f:
        hash_obj = hashlib.new(algorithm)
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REANKER_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "Qwen3-VL-Reranker-2B")


class AssetVectorDBVL(AssetVectorDB):
    """Asset vector database management class (based on ChromaDB)"""

    def __init__(
        self,
        config: Dict,
        db_path: str = CHROMA_DB_CACHE_PATH,
        auto_sync: bool = True,
        force_rebuild: bool = False,
    ):
        """
        Initialize asset vector database

        Args:
            config: Configuration dictionary containing model, API and other information
            db_path: Database storage path, defaults to current directory
            auto_sync: Whether to automatically sync YAML files
        """
        # Initialize configuration
        self.config = config
        self.use_reranker = config.get("use_reranker", False)
        self.permanent_model_in_gpu = config.get("permanent_model_in_gpu", False)
        self.model_path = config.get("model", "Qwen3-VL-Embedding-2B")
        self.model_path = os.path.join(CURRENT_DIR, "models", self.model_path)

        # Set database path
        if db_path is None:
            db_path = CHROMA_DB_CACHE_PATH
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        if not os.path.exists(db_path) or not os.path.exists(f"{db_path}/assets_sync_state.json") and not force_rebuild:
            if os.path.exists(db_path):
                shutil.rmtree(db_path, ignore_errors=True)
            logger.info(f"Creating ChromaDB cache directory: {db_path}")
            os.makedirs(db_path, exist_ok=True, mode=0o777)
        else:
            logger.info(f"ChromaDB cache directory already exists: {db_path}")

        # if db_path is readonly, change it to writable
        if not os.access(db_path, os.W_OK):
            os.chmod(db_path, 0o777)
            logger.info(f"Changed ChromaDB cache directory to writable: {db_path}")

        self.assets_index = ASSETS_INDEX
        # Define file paths
        self.db_path = db_path
        self.chroma_persist_directory = db_path
        self.sync_state_path = f"{db_path}/assets_sync_state.json"
        self.embeddings = QwenVLEmbeddings(
            model_path=self.model_path, permanent_model_in_gpu=self.permanent_model_in_gpu
        )
        self.embedding_model = self.embeddings.model_path
        self.embeddings.validate_environment()

        # Initialize ChromaDB
        self.collection_name = "assets_collection"
        self.vectorstore = None
        self.is_initialized = False

        # Initialize reranker with auto-cleanup
        self.reranker = None
        self._reranker_last_used_time: float = 0.0
        self._reranker_cleanup_thread: threading.Thread = None
        self._reranker_lock: threading.Lock = threading.Lock()
        self._reranker_shutdown: bool = False
        self._idle_clear_time: float = 5.0
        if self.use_reranker:
            if self.permanent_model_in_gpu:
                self._create_reranker()
            else:
                self._start_reranker_cleanup_thread()

        # Auto sync
        if auto_sync:
            self.sync(force_rebuild)
        else:
            self._initialize_chroma()

    def _create_reranker(self):
        """Create reranker instance"""
        if self.reranker is None:
            self.reranker = Qwen3VLReranker(
                model_name_or_path=REANKER_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            logger.info("Reranker created")

    def _clear_reranker(self):
        """Clear reranker to free GPU memory"""
        if self.reranker is not None:
            try:
                # Clear model from GPU
                if hasattr(self.reranker, "model") and self.reranker.model is not None:
                    self.reranker.model = None
                if hasattr(self.reranker, "processor") and self.reranker.processor is not None:
                    self.reranker.processor = None
                if hasattr(self.reranker, "score_linear") and self.reranker.score_linear is not None:
                    self.reranker.score_linear = None
                self.reranker = None
                # Force garbage collection and clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Reranker cleared from GPU memory")
            except Exception as e:
                logger.warning(f"Error clearing reranker: {e}")
                self.reranker = None

    def _start_reranker_cleanup_thread(self):
        """Start background thread to cleanup reranker after idle time"""
        if self._reranker_cleanup_thread is None or not self._reranker_cleanup_thread.is_alive():
            self._reranker_shutdown = False
            self._reranker_cleanup_thread = threading.Thread(target=self._reranker_cleanup_worker, daemon=True)
            self._reranker_cleanup_thread.start()
            logger.info("Reranker cleanup thread started")

    def _reranker_cleanup_worker(self):
        """Background worker to cleanup reranker after idle time"""
        while not self._reranker_shutdown:
            time.sleep(1)  # Check every second
            if self.permanent_model_in_gpu:
                continue
            with self._reranker_lock:
                current_time = time.time()
                if self.reranker is not None and self._reranker_last_used_time > 0:
                    idle_time = current_time - self._reranker_last_used_time
                    if idle_time >= self._idle_clear_time:
                        logger.info(f"Reranker idle for {idle_time:.2f}s, clearing from GPU")
                        self._clear_reranker()

    def _ensure_reranker(self):
        """Ensure reranker exists, create if needed"""
        if self.reranker is None:
            self._create_reranker()
        self._reranker_last_used_time = time.time()

    def _get_asset_video_path_from_id(self, asset_id: str) -> str:
        asset_info = self.assets_index.get(asset_id, None)
        if not asset_info:
            return None
        return self._get_asset_video_path(asset_info)

    def _get_asset_video_path(self, asset_info: Dict) -> str:
        if asset_info.get("video", "") != "" and os.path.exists(asset_info.get("video")):
            return asset_info.get("video")
        asset_url = asset_info.get("url", None)
        if not asset_url:
            return None
        asset_dir = os.path.dirname(os.path.join(ASSETS_PATH, asset_url))
        asset_video_path = os.path.join(asset_dir, "video/merged.mp4")
        if not os.path.exists(asset_video_path):
            return None
        return asset_video_path

    def _load_assets(self) -> List[Dict]:
        assets_data = []
        for asset_id, asset_info in self.assets_index.items():
            asset_video_path = self._get_asset_video_path(asset_info)
            if not asset_video_path:
                logger.warning(f"Asset {asset_id} video path does not exist, skipping")
                continue
            asset_video_hash = calculate_file_hash(asset_video_path)
            desc = asset_info.get("description", {})
            semantic_names = desc.get("semantic_name", [])
            full_descriptions = desc.get("full_description", [])
            semantic_name = ""
            if semantic_names:
                if isinstance(semantic_names, list):
                    semantic_name = semantic_names[0] if semantic_names else ""
                else:
                    semantic_name = str(semantic_names)
            full_description = ""
            if full_descriptions:
                if isinstance(full_descriptions, list):
                    full_description = full_descriptions[0] if full_descriptions else ""
                else:
                    full_description = str(full_descriptions)

            if not semantic_name and not full_description:
                logger.warning(f"Asset {asset_id} missing semantic name or full description, skipping")
                continue

            # Build asset data
            asset_data = {
                "asset_id": asset_id,
                "video_path": asset_video_path,
                "video_hash": asset_video_hash,
                "semantic_description": f"{semantic_name}: {full_description}",
                "semantic_name": semantic_name,
            }

            # Preserve all other information
            for key, value in asset_info.items():
                if key not in ["video_path", "video_hash", "semantic_description", "asset_id"]:
                    asset_data[key] = value

            assets_data.append(asset_data)

        logger.info(f"Loaded {len(assets_data)} assets")
        return assets_data

    def _create_document(self, asset_info: Dict) -> Document:
        """Create Document object from asset information"""
        asset_id = asset_info.get("asset_id", "")
        semantic_name = asset_info.get("semantic_name", "")
        semantic_description = asset_info.get("semantic_description", "")
        video_path = asset_info.get("video_path", "")
        video_hash = asset_info.get("video_hash", "")
        if not asset_id or not semantic_description or not video_path:
            logger.warning(f"Asset {asset_id} missing required fields, skipping")
            return None

        # Keep original text structure: "name: description"
        page_json = {
            "video": video_path,
            "semantic_name": semantic_name,
            "semantic_description": semantic_description,
        }

        page_content = json.dumps(page_json)
        # Prepare metadata
        metadata = {
            "asset_id": asset_id,
            "semantic_description": semantic_description,
            "video_path": video_path,
            "video_hash": video_hash,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        # metadata.update(asset_info)

        return Document(page_content=page_content, metadata=metadata)

    def sync(self, force_rebuild: bool = False) -> Dict:
        """
        Sync vector database with assets files
        """
        # Load sync state
        sync_state = self._load_sync_state()
        current_assets_hash = ASSETS_INDEX_HASH

        self._initialize_chroma()
        if self.count() == 0:
            logger.warning("ChromaDB is empty, will rebuild")
            force_rebuild = True
        # Check if sync is needed
        if (
            not force_rebuild
            and sync_state.get("assets_hash") == current_assets_hash
            and sync_state.get("last_sync_time")
        ):
            logger.info("Assets file unchanged, skipping sync")
            return {
                "synced": False,
                "reason": "no_changes",
                "asset_count": self.count(),
            }

        logger.info("Assets file changed, starting sync")
        if force_rebuild:
            self.vectorstore.reset_collection()
            logger.info(f"Vector database reset, current asset count: {self.count()}")

        # Load assets
        assets = self._load_assets()
        # Delete assets not in current assets list
        existing_assets = self.get_all()
        assets_id_list = [asset["asset_id"] for asset in assets]
        for existing_asset in existing_assets:
            if existing_asset["asset_id"] not in assets_id_list:
                self.vectorstore._collection.delete(where={"asset_id": existing_asset["asset_id"]})
                logger.info(f"Asset deleted: {existing_asset['asset_id']}")
        # Delete changed assets
        if assets:
            to_be_added = []
            for asset in assets:
                in_db_asset = self._search_by_asset_id(asset["asset_id"], top_k=1, from_assets_dict=False)
                if in_db_asset:
                    asset_video_hash = asset.get("video_hash", "")
                    if "video_hash" not in in_db_asset[0] or in_db_asset[0].get("video_hash") != asset_video_hash:
                        self.vectorstore._collection.delete(where={"asset_id": asset["asset_id"]})
                        logger.info(f"Asset changed: {asset['asset_id']}")
                        to_be_added.append(asset)
                else:
                    to_be_added.append(asset)
                    logger.info(f"New asset found: {asset['asset_id']}")
            if to_be_added:
                doc_ids = []
                documents = []
                for asset in to_be_added:
                    doc = self._create_document(asset)
                    if doc:
                        documents.append(doc)
                        doc_ids.append(asset["asset_id"])
                logger.info(f"Adding {len(documents)} assets to ChromaDB")
                self.embeddings.permanent_model_in_gpu = True
                self.embeddings.batch_size = int(os.getenv("BATCH_SIZE", 10))
                self.vectorstore.add_documents(documents=documents, ids=doc_ids)
                self.embeddings.permanent_model_in_gpu = self.permanent_model_in_gpu

                logger.info(f"Added {len(documents)} assets to ChromaDB")

        # Update sync state
        new_sync_state = {
            "last_sync_time": datetime.now().isoformat(),
            "assets_hash": current_assets_hash,
            "asset_count": self.count(),
        }
        self._save_sync_state(new_sync_state)

        logger.info(f"Sync completed, current asset count: {self.count()}")
        return {
            "synced": True,
            "asset_count": self.count(),
            "last_sync_time": new_sync_state["last_sync_time"],
        }

    def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 10,
        exclude_regex: str = None,
        include_regex: str = None,
        scene_description: str = None,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """
        Search for similar assets

        Args:
            query: Query text, can be a single string or a list of strings
            top_k: Return the top k most similar results

        Returns:
            Search results
        """
        if not self.use_reranker:
            return super().search(query, top_k, exclude_regex, include_regex, scene_description)
        serarch_top_k = max(top_k * 2, 10)
        tmp_results = super().search(query, serarch_top_k, exclude_regex, include_regex, scene_description)

        logger.info("Reranking...=========================")
        results = []
        if isinstance(query, str):
            query_list = [query]
            tmp_results = [tmp_results]
        else:
            query_list = query
        for i, query_results in enumerate(tmp_results):
            documents = []
            for query_result in query_results:
                asset_id = query_result["asset_id"]
                asset_video_path = self._get_asset_video_path_from_id(asset_id)
                asset_semantic_description = (
                    self.assets_index.get(asset_id, {}).get("description", {}).get("full_description", "")
                )
                documents.append({"video": asset_video_path, "text": asset_semantic_description})
            inputs = {
                "instruction": f"The user could provide a description of a scene along with the keyword(query) to  search. The keyword should describe an object in the scene. You need to retrieve the most relevant multi-view video of an asset from the documents, to answer the user's query.The user's scene description is: {scene_description}.",
                "query": {"text": query_list[i]},
                "documents": documents,
                "fps": 3.0,
            }
            with self._reranker_lock:
                self._ensure_reranker()
                reranker_scores = self.reranker.process(inputs)
                self._ensure_reranker()
            # Ensure the length of reranker_scores matches the number of query_results
            if len(query_results) != len(reranker_scores):
                logger.warning(
                    f"Length mismatch: {len(query_results)} query_results, " f"{len(reranker_scores)} reranker scores"
                )
                # Truncate or pad reranker_scores if needed (avoid crash)
                min_len = min(len(query_results), len(reranker_scores))
                query_results = query_results[:min_len]
                reranker_scores = reranker_scores[:min_len]
            # Combine results and scores
            results_with_scores = [(qr, float(score)) for qr, score in zip(query_results, reranker_scores)]
            # Sort by scores in descending order
            sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
            # Print out sorted results
            results.append([qr for qr, score in sorted_results])
        # for i in range(len(query_list)):
        #     logger.info(f"\nQuery '{query_list[i]}' results:")
        #     query_results_origin = tmp_results[i]
        #     query_results_reranked = results[i]
        #     for j, result in enumerate(query_results_origin):
        #         logger.info(f"  {j+1}. {result['asset_id']}...")
        #     logger.info(f"Reranked results:")
        #     for j, result in enumerate(query_results_reranked):
        #         logger.info(f"  {j+1}. {result['asset_id']}")
        if isinstance(query, str):
            return results[0][:top_k]
        else:
            return [result[:top_k] for result in results]


def use_example():
    # Example configuration
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    # Create database instance (auto sync YAML)
    logger.info("=== Creating database instance ===")
    config = {"use_reranker": True}
    db = AssetVectorDBVL(config=config, auto_sync=True, force_rebuild=False)
    # queries = ["yellow brick", "star-shaped brick", "bottle", "freezer", "furniture", "wine", "dish", "cola", "sour plum drink", "fast food", "white sugar"]
    queries = ["bottle"]
    start_time = time.time()
    batch_results = db.search(
        queries,
        top_k=10,
        exclude_regex="omni6D",
        scene_description="The kitchen counter has many bottled condiments: soy sauce, vinegar, cooking wine, sugar, salt, pepper, chili sauce, etc.",
    )
    end_time = time.time()
    logger.info(f"Search time: {end_time - start_time} seconds")
    for i, query_results in enumerate(batch_results):
        logger.info(f"\nQuery '{queries[i]}' results:")
        for j, result in enumerate(query_results):
            logger.info(f"  {j+1}. {result['asset_id']}...")

    # time.sleep(6)
    # batch_results = db.search(queries, top_k=5)
    # for i, query_results in enumerate(batch_results):
    #     logger.info(f"\nQuery '{queries[i]}' results:")
    #     for j, result in enumerate(query_results):
    #         logger.info(f"  {j+1}. {result['asset_id']}...")


if __name__ == "__main__":
    use_example()
