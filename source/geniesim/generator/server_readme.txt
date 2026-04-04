Choose one of the following two options:

    1. qwen embedding API:
        - Configure source/geniesim/generator/server/mcp_text_embedding/text_embedding_config.json
        - From $generator: docker compose --profile text up --build

    2. qwen embedding model (requires GPU and Nvidia Container Toolkit):
        - cd server/assets_searcher
        - bash download_models.sh  # Downloads model and scripts; see download_model.sh for details

        - Adjust BATCH_SIZE and USE_RERANKER under services/mcp-server_vl/environment in source/geniesim/generator/compose.yaml for your hardware:
            - BATCH_SIZE: Batch size when building the vector index; larger uses more VRAM, smaller takes longer. Default 10 (e.g. RTX 4090).
            - USE_RERANKER: Whether to use the reranker (True/False). Reranking improves search quality but increases latency and VRAM use.
        - From $generator: docker compose --profile vl up --build
