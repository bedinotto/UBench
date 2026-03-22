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

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed!${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_CMD="python3"
echo -e "${GREEN}✓${NC} Python 3 found: $($PYTHON_CMD --version)"

# Function to run setup
run_setup() {
    echo ""
    echo "================================================================================"
    echo "STEP 1: ENVIRONMENT SETUP"
    echo "================================================================================"
    echo ""
    
    $PYTHON_CMD codes/setup.py
    
    if [ $? -ne 0 ]; then
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
    
    # Pass all arguments to the pipeline
    $PYTHON_CMD codes/main_pipeline.py "$@"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Pipeline failed!${NC}"
        exit 1
    fi
}

# Main execution
main() {
    # Check if --skip-setup flag is provided
    SKIP_SETUP=false
    for arg in "$@"; do
        if [ "$arg" == "--skip-setup" ]; then
            SKIP_SETUP=true
            # Remove --skip-setup from arguments
            set -- "${@/$arg/}"
        fi
    done
    
    # Run setup unless skipped
    if [ "$SKIP_SETUP" = false ]; then
        run_setup
    else
        echo -e "${YELLOW}Skipping setup (--skip-setup flag detected)${NC}"
    fi
    
    # Run pipeline with remaining arguments
    run_pipeline "$@"
    
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}✓✓✓ PIPELINE COMPLETED SUCCESSFULLY ✓✓✓${NC}"
    echo "================================================================================"
    echo ""
    echo "Results are available in:"
    echo "  - outputs/models/     (trained model weights)"
    echo "  - outputs/plots/      (training curves and comparisons)"
    echo "  - log/                (training logs and metrics)"
    echo ""
    echo "================================================================================"
}

# Show usage if --help is requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-setup              Skip environment setup (use if already set up)"
    echo "  --models MODEL [MODEL...] Train only specific models (unet, transunet, swin)"
    echo "  --skip-benchmark          Skip benchmarking after training"
    echo "  --epochs N                Number of training epochs (default: 100)"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                                  # Full pipeline with setup"
    echo "  ./run.sh --skip-setup                     # Run without setup"
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
