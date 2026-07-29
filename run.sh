#!/bin/bash
# =============================================================================
# Automated Setup and Training Pipeline - Linux/Mac
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "THERMAL FACE DETECTION - AUTOMATED PIPELINE"
echo "================================================================================"
echo ""

PYTHON_CMD="python3"

# 1. Install uv automatically if it isn't installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Tell uv to automatically fetch Python 3.11 and create the .venv
uv venv --python 3.11 .venv

# 3. Activate the environment
source .venv/bin/activate

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed!${NC}"
    echo "Please install UV to install Python 3.11 - see https://astral.sh/uv/install.sh"
    exit 1
fi

# Look for specific compatible versions first
if command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &>/dev/null; then
    PYTHON_CMD="python3.10"
fi

$PYTHON_CMD -m venv .venv
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Python 3 found: $($PYTHON_CMD --version)"

export NO_ALBUMENTATIONS_UPDATE=1

# Create a timestamp for this run and share it with the Python pipeline so
# shell and Python use ONE run id — one outputs/<ts> and one logs/<ts> per
# invocation (UB-14). Precedence in Python: --resume > UBENCH_RUN_ID > fresh.
RUN_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
export UBENCH_RUN_ID="${RUN_TIMESTAMP}"
RUN_LOG_DIR="logs/${RUN_TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"
echo -e "${GREEN}✓${NC} Run ID: ${RUN_TIMESTAMP}"

# Function to extract data
run_extract() {
    echo ""
    echo "================================================================================"
    echo "STEP 0: DATA EXTRACTION"
    echo "================================================================================"
    echo ""

    $PYTHON_CMD -m codes.extract_data 2>&1 | tee -a "${RUN_LOG_DIR}/extract.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}ERROR: Data extraction failed!${NC}"
        exit 1
    fi
}

# Function to run setup
run_setup() {
    echo ""
    echo "================================================================================"
    echo "STEP 1: ENVIRONMENT SETUP"
    echo "================================================================================"
    echo ""

    $PYTHON_CMD -m codes.setup 2>&1 | tee -a "${RUN_LOG_DIR}/setup.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}ERROR: Setup failed!${NC}"
        exit 1
    fi
}

# Function to run pipeline
run_pipeline() {
    echo ""
    echo "================================================================================"
    echo "STEP 2: RUNNING TRAINING PIPELINE"
    echo "================================================================================"
    echo ""
    
    # Pass all arguments to the pipeline. Python's TeeLogger owns
    # ${RUN_LOG_DIR}/pipeline.log (same dir now that the run id is shared);
    # the shell captures its own view in console.log so two writers never
    # share one file.
    $PYTHON_CMD -m codes.main_pipeline "$@" 2>&1 | tee -a "${RUN_LOG_DIR}/console.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}ERROR: Pipeline failed!${NC}"
        exit 1
    fi
}

# Main execution
main() {
    # Parse flags
    SKIP_SETUP=false
    SKIP_EXTRACT=false
    PIPELINE_ARGS=()
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --skip-extract)
                SKIP_EXTRACT=true
                shift
                ;;
            *)
                PIPELINE_ARGS+=("$1")
                shift
                ;;
        esac
    done
    
    # Run data extraction unless skipped
    if [ "$SKIP_EXTRACT" = false ]; then
        run_extract
    else
        echo -e "${YELLOW}Skipping data extraction (--skip-extract flag detected)${NC}"
    fi
    
    # Run setup unless skipped
    if [ "$SKIP_SETUP" = false ]; then
        run_setup
    else
        echo -e "${YELLOW}Skipping setup (--skip-setup flag detected)${NC}"
    fi
    
    # Run pipeline with remaining arguments
    run_pipeline "${PIPELINE_ARGS[@]}"
    
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✓✓✓ PIPELINE COMPLETED SUCCESSFULLY ✓✓✓${NC}"
    echo "================================================================================"
    echo ""
    echo "Run ID: ${RUN_TIMESTAMP}"
    echo "Results are available in timestamped subdirectories:"
    echo "  - outputs/<timestamp>/models/   (trained model weights)"
    echo "  - outputs/<timestamp>/plots/    (training curves and comparisons)"
    echo "  - logs/<timestamp>/             (training logs and metrics)"
    echo ""
    echo "Full console log: ${RUN_LOG_DIR}/pipeline.log"
    echo ""
    echo "================================================================================"
}

# Show usage if --help is requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-setup              Skip environment setup (use if already set up)"
    echo "  --skip-extract            Skip data extraction from ZIP files"
    echo "  --models MODEL [MODEL...] Train only specific models (unet, transunet, swin)"
    echo "  --skip-benchmark          Skip benchmarking after training"
    echo "  --epochs N                Number of training epochs (default: from codes/config.yaml)"
    echo "  --fail-fast               Abort on the first model/fold training failure"
    echo "  --resume RUN_ID           Reuse outputs/<RUN_ID> and logs/<RUN_ID> and continue"
    echo "                            from its checkpoints (overrides this script's run id)"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                                  # Full pipeline with setup"
    echo "  ./run.sh --skip-setup                     # Run without setup"
    echo "  ./run.sh --skip-extract --skip-setup      # Skip extraction and setup"
    echo "  ./run.sh --models unet                    # Train only U-Net"
    echo "  ./run.sh --models transunet swin          # Train TransUNet and Swin-UNet++"
    echo "  ./run.sh --skip-benchmark                 # Train all, skip benchmark"
    echo "  ./run.sh --epochs 50                      # Train for 50 epochs"
    echo "  ./run.sh --skip-setup --models unet       # Quick U-Net training"
    echo ""
    exit 0
fi

# Run main function
main "$@"
