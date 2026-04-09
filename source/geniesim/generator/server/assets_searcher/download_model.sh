#!/bin/bash

# Model download script
# Args: 2B, 8B, or ALL (default 2B)
# Pass --huggingface to use Hugging Face; otherwise ModelScope
# Model IDs: qwen/Qwen3-VL-Embedding-2B qwen/Qwen3-VL-Embedding-8B qwen/Qwen3-VL-Reranker-2B qwen/Qwen3-VL-Reranker-8B
# Local paths: ./models/Qwen3-VL-Embedding-2B ./models/Qwen3-VL-Embedding-8B ./models/Qwen3-VL-Reranker-2B ./models/Qwen3-VL-Reranker-8B
# Sizes: 2B 8B 2B 8B

# Print help
show_help() {
    echo "Usage: $0 [2B|8B|ALL] [--huggingface]"
    echo ""
    echo "Arguments:"
    echo "  2B           Download 2B models (Embedding-2B and Reranker-2B) [default]"
    echo "  8B           Download 8B models (Embedding-8B and Reranker-8B)"
    echo "  ALL          Download all model sizes"
    echo "  --huggingface Use Hugging Face (default: ModelScope)"
    echo "  -h, --help    Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                    # ModelScope, 2B"
    echo "  $0 8B                 # ModelScope, 8B"
    echo "  $0 ALL --huggingface  # Hugging Face, all sizes"
}

# Parse arguments
VERSION="2B"
USE_HUGGINGFACE=false

for arg in "$@"; do
    case $arg in
        2B|8B|ALL)
            VERSION="$arg"
            ;;
        --huggingface)
            USE_HUGGINGFACE=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$arg'"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

for arg in "$@"; do
    case $arg in
        2B|8B|ALL)
            VERSION="$arg"
            ;;
        --huggingface)
            USE_HUGGINGFACE=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [2B|8B|ALL] [--huggingface]"
            exit 1
            ;;
    esac
done

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

# Ensure models directory exists
mkdir -p "${MODELS_DIR}"

# Download one model
download_model() {
    local model_name=$1
    local model_path=$2

    echo "Downloading model: $model_name -> $model_path"

    if [ "$USE_HUGGINGFACE" = true ]; then
        # huggingface-cli
        if command -v huggingface-cli &> /dev/null; then
            huggingface-cli download "$model_name" --local-dir "$model_path" --local-dir-use-symlinks False
        else
            echo "Error: huggingface-cli not found; install with: pip install huggingface_hub[cli]"
            exit 1
        fi
    else
        # modelscope
        if command -v modelscope &> /dev/null; then
            # Parent of target must exist
            local parent_dir=$(dirname "$model_path")
            mkdir -p "$parent_dir"

            modelscope download --model "$model_name" --local_dir "$model_path"

            if [ $? -eq 0 ]; then
                echo "✓ Model downloaded: $model_path"
            else
                echo "✗ Model download failed: $model_path"
                exit 1
            fi
        else
            echo "Error: modelscope not found; install with: pip install modelscope"
            exit 1
        fi
    fi
}

# Models to fetch for this VERSION
declare -a MODELS_TO_DOWNLOAD

if [ "$VERSION" = "2B" ]; then
    MODELS_TO_DOWNLOAD=(
        "qwen/Qwen3-VL-Embedding-2B:${MODELS_DIR}/Qwen3-VL-Embedding-2B"
        "qwen/Qwen3-VL-Reranker-2B:${MODELS_DIR}/Qwen3-VL-Reranker-2B"
    )
elif [ "$VERSION" = "8B" ]; then
    MODELS_TO_DOWNLOAD=(
        "qwen/Qwen3-VL-Embedding-8B:${MODELS_DIR}/Qwen3-VL-Embedding-8B"
        "qwen/Qwen3-VL-Reranker-8B:${MODELS_DIR}/Qwen3-VL-Reranker-8B"
    )
elif [ "$VERSION" = "ALL" ]; then
    MODELS_TO_DOWNLOAD=(
        "qwen/Qwen3-VL-Embedding-2B:${MODELS_DIR}/Qwen3-VL-Embedding-2B"
        "qwen/Qwen3-VL-Embedding-8B:${MODELS_DIR}/Qwen3-VL-Embedding-8B"
        "qwen/Qwen3-VL-Reranker-2B:${MODELS_DIR}/Qwen3-VL-Reranker-2B"
        "qwen/Qwen3-VL-Reranker-8B:${MODELS_DIR}/Qwen3-VL-Reranker-8B"
    )
fi

# Download models
echo "========================================="
echo "Starting download (version: $VERSION, backend: $([ "$USE_HUGGINGFACE" = true ] && echo "HuggingFace" || echo "ModelScope"))"
echo "========================================="

for model_info in "${MODELS_TO_DOWNLOAD[@]}"; do
    IFS=':' read -r model_name model_path <<< "$model_info"
    download_model "$model_name" "$model_path"
done

# Python helper scripts from upstream
echo ""
echo "========================================="
echo "Downloading Python helper scripts"
echo "========================================="

download_python_file() {
    local url=$1
    local output_path=$2
    local filename=$(basename "$output_path")

    echo "Downloading: $filename"

    if command -v curl &> /dev/null; then
        curl -L -o "$output_path" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$output_path" "$url"
    else
        echo "Error: neither curl nor wget found"
        exit 1
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Saved: $output_path"
    else
        echo "✗ Download failed: $output_path"
        exit 1
    fi
}

# qwen3_vl_embedding.py
EMBEDDING_URL="https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/main/src/models/qwen3_vl_embedding.py"
EMBEDDING_PATH="${MODELS_DIR}/qwen3_vl_embedding.py"
download_python_file "$EMBEDDING_URL" "$EMBEDDING_PATH"

# qwen3_vl_reranker.py
RERANKER_URL="https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/main/src/models/qwen3_vl_reranker.py"
RERANKER_PATH="${MODELS_DIR}/qwen3_vl_reranker.py"
download_python_file "$RERANKER_URL" "$RERANKER_PATH"

echo ""
echo "========================================="
echo "All downloads finished."
echo "========================================="
