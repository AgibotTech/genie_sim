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
from typing import List, Dict, Union, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from geniesim.generator.server.text_embedding import DashscopeTextEmbeddings, TextEmbeddings
from geniesim.assets import (
    ASSETS_INDEX,
    ASSETS_INDEX_HASH,
    CHROMA_DB_PATH,
)  # If this fails please check assets folder at source/geniesim/assets

CHROMA_DB_CACHE_PATH = os.path.join("/tmp", "chromadb_cache")

import shutil

# Configure logging to use stderr to avoid interfering with JSON-RPC on stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class AssetVectorDB:
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
        self.embedding_model = config.get("model", "text-embedding-v4")
        self.api_key = config.get("api_key", None)
        self.base_url = config.get("base_url", None)
        self.dimension = config.get("dimension", 2048)
        self.dashscope_mode = config.get("dashscope_mode", True)

        # Set database path
        if db_path is None:
            db_path = CHROMA_DB_CACHE_PATH
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        if not os.path.exists(CHROMA_DB_PATH):
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            force_rebuild = True
            logger.info(f"CHROMA_DB_PATH does not exist, will rebuild")
        if not os.path.exists(db_path) or not os.path.exists(f"{db_path}/assets_sync_state.json") and not force_rebuild:
            if os.path.exists(db_path):
                shutil.rmtree(db_path, ignore_errors=True)
            logger.info(f"Creating ChromaDB cache directory: {db_path}")
            os.makedirs(db_path, exist_ok=True, mode=0o777)
            logger.info(f"Copying ChromaDB cache directory: {CHROMA_DB_PATH} to {db_path}")
            shutil.copytree(CHROMA_DB_PATH, db_path, dirs_exist_ok=True)
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
        if self.dashscope_mode:
            self.embeddings = DashscopeTextEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key,
                dimension=self.dimension,
            )
        else:
            self.embeddings = TextEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key,
                base_url=self.base_url,
                dimension=self.dimension,
            )
        self.embeddings.validate_environment()

        # Initialize ChromaDB
        self.collection_name = "assets_collection"
        self.vectorstore = None
        self.is_initialized = False

        # Auto sync
        if auto_sync:
            self.sync(force_rebuild)
        else:
            self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB"""
        try:
            logger.info(f"Initializing ChromaDB, persist directory: {self.chroma_persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                collection_configuration={"hnsw": {"space": "cosine"}},
            )

            self.is_initialized = True
            logger.info(f"ChromaDB initialized successfully, current asset count: {self.count()}")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            logger.error(traceback.format_exc())
            self.is_initialized = False

    def _load_sync_state(self) -> Dict:
        """Load sync state"""
        if os.path.exists(self.sync_state_path):
            try:
                with open(self.sync_state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"last_sync_time": None, "assets_hash": None}

    def _save_sync_state(self, state: Dict):
        """Save sync state"""
        os.makedirs(os.path.dirname(self.sync_state_path), exist_ok=True)
        with open(self.sync_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_assets(self) -> List[Dict]:
        assets_data = []
        for asset_id, asset_info in self.assets_index.items():
            desc = asset_info.get("description", {})
            semantic_names = desc.get("semantic_name", [])
            full_descriptions = desc.get("full_description", [])

            # Handle possible list or string format
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
                "semantic_description": f"{semantic_name}: {full_description}",
            }

            # Preserve all other information
            for key, value in asset_info.items():
                if key not in ["semantic_description", "asset_id"]:
                    asset_data[key] = value

            assets_data.append(asset_data)

        logger.info(f"Loaded {len(assets_data)} assets")
        return assets_data

    def _create_document(self, asset_info: Dict) -> Document:
        """Create Document object from asset information"""
        asset_id = asset_info.get("asset_id", "")
        semantic_description = asset_info.get("semantic_description", "")
        if not asset_id or not semantic_description:
            logger.warning(f"Asset {asset_id} missing required fields, skipping")
            return None

        # Keep original text structure: "name: description"
        page_content = semantic_description
        # Prepare metadata
        metadata = {
            "asset_id": asset_id,
            "semantic_description": semantic_description,
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
                in_db_asset = self._search_by_asset_id(asset["asset_id"], top_k=1)
                if in_db_asset:
                    asset_semantic_description = asset.get("semantic_description", "")
                    if in_db_asset[0].get("semantic_description") != asset_semantic_description:
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
                self.vectorstore.add_documents(documents=documents, ids=doc_ids)

                logger.info(f"Added {len(documents)} assets to ChromaDB")

        # Update sync state
        new_sync_state = {
            "last_sync_time": datetime.now().isoformat(),
            "assets_hash": current_assets_hash,
            "asset_count": self.count(),
        }
        self._save_sync_state(new_sync_state)

        logger.info(f"Sync completed, current asset count: {self.count()}")
        logger.info(f"Syncing back to chromadb directory: {CHROMA_DB_PATH}")
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        # shutil.copytree(self.chroma_persist_directory, CHROMA_DB_PATH, dirs_exist_ok=True)
        return {
            "synced": True,
            "asset_count": self.count(),
            "last_sync_time": new_sync_state["last_sync_time"],
        }

    def search(self, query: Union[str, List[str]], top_k: int = 10) -> Union[List[Dict], List[List[Dict]]]:
        """
        Search for similar assets

        Args:
            query: Query text, can be a single string or a list of strings
            top_k: Return the top k most similar results

        Returns:
            Search results
        """
        before = time.time()
        if not self.embedding_model:
            raise ValueError("Text embedding model or base_url not set, please check configuration")

        if not self.is_initialized or not self.vectorstore:
            logger.warning("Vector database not initialized or empty")
            return [] if isinstance(query, str) else [[]]

        # Process query input
        is_single_query = isinstance(query, str)
        queries = [query] if is_single_query else query

        all_results = []

        for query_text in queries:
            try:
                query_results = []
                # First search by ID
                id_results = self._search_by_asset_id(query_text, top_k=top_k)
                for id_result in id_results:
                    asset_id = id_result.get("asset_id")
                    asset_params = self.assets_index.get(asset_id, None)
                    if not asset_params:
                        continue
                    result = {
                        "asset_id": asset_id,
                        "description": asset_params.get("description", {}),
                    }
                    query_results.append(result)
                if len(query_results) >= top_k:
                    query_results = query_results[:top_k]
                else:
                    # Use ChromaDB similarity search
                    docs = self.vectorstore.similarity_search_with_relevance_scores(
                        query=query_text, k=min(top_k, self.count())
                    )
                    for doc, score in docs:
                        metadata = doc.metadata.copy()
                        asset_id = metadata.get("asset_id")
                        if not asset_id:
                            continue
                        already_in_query = False
                        for query_result in id_results:
                            if query_result.get("asset_id") == asset_id:
                                already_in_query = True
                                break
                        if already_in_query:
                            continue
                        asset_params = self.assets_index.get(asset_id, None)
                        if not asset_params or "description" not in asset_params:
                            continue
                        result = {
                            "asset_id": asset_id,
                            "description": asset_params.get("description"),
                        }
                        query_results.append(result)
                        if len(query_results) >= top_k:
                            break

                all_results.append(query_results)

            except Exception as e:
                logger.error(f"Error searching query '{query_text}': {e}")
                all_results.append([])

        # Format results
        if isinstance(query, str):
            logger.info(f"Search '{query}' returned {len(all_results[0])} results")
            after = time.time()
            logger.info(f"Search took {after - before:.2f} seconds")
            return all_results[0]
        else:
            logger.info(f"Batch search {len(query)} queries, each returning up to {top_k} results")
            after = time.time()
            logger.info(f"Search took {after - before:.2f} seconds")
            return all_results

    def _search_by_asset_id(self, asset_id: str, top_k: int = 1, from_assets_dict=True) -> List[Dict]:
        """Search for asset by asset_id"""
        if from_assets_dict:
            asset_params = self.assets_index.get(asset_id, None)
            if not asset_params:
                return []
            result = {
                "asset_id": asset_id,
                "description": asset_params.get("description", {}),
            }
            return [result]
        try:
            docs = self.vectorstore._collection.get(where={"asset_id": asset_id}, limit=top_k)

            if docs and "metadatas" in docs and docs["metadatas"]:
                results = []
                for metadata in docs["metadatas"][:top_k]:
                    # Build return structure
                    result = metadata
                    results.append(result)

                return results
        except Exception as e:
            logger.error(f"Error searching by asset_id: {e}")

        return []

    def search_by_asset_id(self, asset_id: str, top_k: int = 1) -> List[Dict]:
        results = self._search_by_asset_id(asset_id, top_k=top_k)
        results_output = []
        for id_result in results:
            asset_id = id_result.get("asset_id")
            asset_params = self.assets_index.get(asset_id, None)
            if not asset_params:
                continue
            result = {
                "asset_id": asset_id,
                "description": asset_params.get("description", {}),
            }
            results_output.append(result)
        return results_output

    def get(self, asset_id: str) -> Optional[Dict]:
        """
        Get asset information by asset_id

        Args:
            asset_id: Asset ID

        Returns:
            Asset information dictionary, returns None if not found
        """
        results = self._search_by_asset_id(asset_id, top_k=1)
        return results[0] if results else None

    def get_all(self) -> List[Dict]:
        """
        Get all asset information
        """
        if not self.is_initialized:
            self._initialize_chroma()
            if not self.is_initialized:
                return []

        try:
            # Get all documents
            docs = self.vectorstore._collection.get()

            if docs and "metadatas" in docs:
                assets = []
                for metadata in docs["metadatas"]:
                    # Build return structure
                    asset_data = metadata
                    assets.append(asset_data)

                return assets
        except Exception as e:
            logger.error(f"Error getting all assets: {e}")

        return []

    def count(self) -> int:
        """
        Get asset count

        Returns:
            Asset count
        """
        if not self.is_initialized or not self.vectorstore:
            return 0

        try:
            # Get document count in collection
            collection = self.vectorstore._collection
            if collection:
                return collection.count()
        except:
            pass

        return 0

    def export_to_list(self, output_path: str = None) -> bool:

        if not self.is_initialized:
            self._initialize_chroma()
            if not self.is_initialized:
                logger.error("Unable to load vector database")
                return False

        # Get all assets
        all_assets = self.get_all()
        if not all_assets:
            logger.warning("No assets to export")
            return False

        # Convert to assets compatible format
        asset_list = []
        for asset in all_assets:
            asset_id = asset.get("asset_id", "")
            if not asset_id:
                continue
            asset_text = f"{asset.get('semantic_description', '')}; asset_id:{asset_id}"
            # Build assets structure
            asset_list.append(asset_text)

        os.makedirs(output_path, exist_ok=True)
        for i, asset in enumerate(asset_list):
            with open(os.path.join(output_path, f"asset_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(asset)
        return True

    def clear_cache(self):
        """Delete cache files"""
        import shutil

        cache_dirs = [self.chroma_persist_directory]
        cache_files = [self.sync_state_path]

        for dir_path in cache_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"Deleted directory: {dir_path}")
                else:
                    logger.info(f"Directory does not exist: {dir_path}")
            except Exception as e:
                logger.error(f"Error deleting directory {dir_path}: {e}")

        for file_path in cache_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                else:
                    logger.info(f"File does not exist: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        # Reset state
        self.vectorstore = None
        self.is_initialized = False

    def status(self) -> Dict:
        """
        Get database status

        Returns:
            Status information dictionary
        """
        sync_state = self._load_sync_state()

        return {
            "initialized": self.is_initialized,
            "asset_count": self.count(),
            "last_sync_time": sync_state.get("last_sync_time"),
            "model": self.embedding_model,
            "chroma_persist_directory": self.chroma_persist_directory,
            "collection_name": self.collection_name,
        }


def use_example():
    # Example configuration
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    # Create database instance (auto sync YAML)
    logger.info("=== Creating database instance ===")
    config = json.load(open(f"{current_dir}/text_embedding_config.json"))
    db = AssetVectorDB(config=config, auto_sync=True, force_rebuild=False)
    queries = ["table_001", "table"]
    batch_results = db.search(queries, top_k=10)

    for i, query_results in enumerate(batch_results):
        logger.info(f"\nQuery '{queries[i]}' results:")
        for j, result in enumerate(query_results):
            logger.info(f"  {j+1}. {result['asset_id']}...")


def force_rebuild():
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    from config.scene_gen_config_loader import SceneGenConfig

    config_path = "config/scene_gen_config.yaml"
    config = SceneGenConfig(config_path=config_path)

    # Create database instance (auto sync YAML)
    logger.info("=== Creating database instance ===")
    db = AssetVectorDB(
        config=config.get_text_embedding_config(),
        auto_sync=True,
    )
    db.sync(force_rebuild=True)


def export():
    from pathlib import Path
    import sys

    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    from config.scene_gen_config_loader import SceneGenConfig

    config_path = "config/scene_gen_config.yaml"
    config = SceneGenConfig(config_path=config_path)
    db = AssetVectorDB(
        config=config.get_text_embedding_config(),
        auto_sync=True,
    )
    db.export_to_list(f"{current_dir}/data_list")


if __name__ == "__main__":
    use_example()
    # force_rebuild()
    # export()
