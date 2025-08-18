#!/bin/bash

# Lightning Prediction Model - Complete Pipeline
# This script runs the entire process from raw data to trained model

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (MODIFY THESE FOR YOUR SYSTEM)
LIGHTNING_DIR="/data/gent/vo/002/gvo00202/vsc46275/lightning_prediction/data/input-LP/observed_lightning/"
CAPE_FILE="/data/gent/vo/002/gvo00202/vsc46275/lightning_prediction/data/input-LP/cape/cape-2019-2022.nc"
TERRAIN_FILE="/data/gent/vo/002/gvo00202/vsc46275/lightning_prediction/data/input-LP/terrain/terrain_lonlat.nc"

# Project directories
PROJECT_ROOT="$(pwd)"
DATA_PROCESSED_DIR="data/processed"
DATA_SPLITS_DIR="data/splits"
EXPERIMENTS_DIR="experiments"
LOGS_DIR="logs"

# Training configuration
EXPERIMENT_NAME="lightning_cape_seasonal_$(date +%Y%m%d_%H%M%S)"
TRAINING_YEARS="2019 2020 2021"
CONFIG_DIR="config"

# Domain bounds for Odisha
DOMAIN_BOUNDS="81.1644 87.52883 17.76351 22.62838"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# FUNCTIONS
# =============================================================================

print_step() {
    echo -e "${BLUE}==== $1 ====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

check_directory_exists() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_success "Created directory: $1"
    fi
}

# =============================================================================
# VALIDATION
# =============================================================================

validate_inputs() {
    print_step "Validating Input Files and Directories"
    
    # Check input files
    check_file_exists "$CAPE_FILE"
    check_file_exists "$TERRAIN_FILE"
    check_directory_exists "$LIGHTNING_DIR"
    
    # Check for lightning files
    if [ -z "$(ls -A $LIGHTNING_DIR/*.nc 2>/dev/null)" ]; then
        print_error "No NetCDF files found in lightning directory: $LIGHTNING_DIR"
        exit 1
    fi
    
    # Check project structure
    if [ ! -f "scripts/preprocess_raw_data.py" ]; then
        print_error "Missing scripts/preprocess_raw_data.py - are you in the project root?"
        exit 1
    fi
    
    # Check Python and required packages
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    print_success "All input validation passed"
}

# =============================================================================
# SETUP
# =============================================================================

setup_directories() {
    print_step "Setting Up Directory Structure"
    
    # Create main directories
    create_directory "$DATA_PROCESSED_DIR"
    create_directory "$DATA_SPLITS_DIR"
    create_directory "$EXPERIMENTS_DIR"
    create_directory "$LOGS_DIR"
    create_directory "outputs"
    create_directory "checkpoints"
    
    # Create data subdirectories
    create_directory "$DATA_PROCESSED_DIR/meteorological/cape"
    create_directory "$DATA_PROCESSED_DIR/lightning"
    create_directory "$DATA_PROCESSED_DIR/terrain"
    
    print_success "Directory structure ready"
}

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

preprocess_data() {
    print_step "Preprocessing Raw Data"
    
    echo "Input files:"
    echo "  Lightning directory: $LIGHTNING_DIR"
    echo "  CAPE file: $CAPE_FILE"
    echo "  Terrain file: $TERRAIN_FILE"
    echo "  Domain bounds: $DOMAIN_BOUNDS"
    echo ""
    
    python scripts/preprocess_raw_data.py \
        --lightning-dir "$LIGHTNING_DIR" \
        --cape-file "$CAPE_FILE" \
        --terrain-file "$TERRAIN_FILE" \
        --output-dir "$DATA_PROCESSED_DIR" \
        --domain $DOMAIN_BOUNDS
    
    if [ $? -eq 0 ]; then
        print_success "Data preprocessing completed"
    else
        print_error "Data preprocessing failed"
        exit 1
    fi
    
    # Show processed data summary
    echo ""
    echo "Processed data summary:"
    find "$DATA_PROCESSED_DIR" -name "*.nc" | wc -l | xargs echo "  Total NetCDF files:"
    find "$DATA_PROCESSED_DIR/meteorological/cape" -name "*.nc" | wc -l | xargs echo "  CAPE files:"
    find "$DATA_PROCESSED_DIR/lightning" -name "*.nc" | wc -l | xargs echo "  Lightning files:"
    find "$DATA_PROCESSED_DIR/terrain" -name "*.nc" | wc -l | xargs echo "  Terrain files:"
}

# =============================================================================
# DATA PREPARATION
# =============================================================================

prepare_data_splits() {
    print_step "Creating Seasonal-Aware Data Splits"
    
    echo "Training years: $TRAINING_YEARS"
    echo "Split strategy: seasonal_aware"
    echo ""
    
    python scripts/prepare_data.py \
        --data-dir "$DATA_PROCESSED_DIR" \
        --output-dir "$DATA_SPLITS_DIR" \
        --split-strategy seasonal_aware \
        --training-years $TRAINING_YEARS \
        --compute-stats \
        --seed 42
    
    if [ $? -eq 0 ]; then
        print_success "Data splits created successfully"
    else
        print_error "Data split creation failed"
        exit 1
    fi
    
    # Show split summary
    echo ""
    echo "Data split summary:"
    if [ -f "$DATA_SPLITS_DIR/train_files.txt" ]; then
        grep -c "^[^#]" "$DATA_SPLITS_DIR/train_files.txt" | xargs echo "  Training files:"
    fi
    if [ -f "$DATA_SPLITS_DIR/val_files.txt" ]; then
        grep -c "^[^#]" "$DATA_SPLITS_DIR/val_files.txt" | xargs echo "  Validation files:"
    fi
    if [ -f "$DATA_SPLITS_DIR/test_files.txt" ]; then
        grep -c "^[^#]" "$DATA_SPLITS_DIR/test_files.txt" | xargs echo "  Test files:"
    fi
    if [ -f "$DATA_SPLITS_DIR/other_years_files.txt" ]; then
        grep -c "^[^#]" "$DATA_SPLITS_DIR/other_years_files.txt" | xargs echo "  Other years files (2022):"
    fi
}

# =============================================================================
# TRAINING
# =============================================================================

train_model() {
    print_step "Training Lightning Prediction Model"
    
    echo "Experiment name: $EXPERIMENT_NAME"
    echo "Configuration: $CONFIG_DIR"
    echo ""
    
    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null; then
        print_success "GPU detected - training will use CUDA"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        print_warning "No GPU detected - training will use CPU (slower)"
    fi
    echo ""
    
    # FIXED: Removed --logger tensorboard --seed 42 arguments
    # These are already configured in the YAML files
    python scripts/train.py \
        --config "$CONFIG_DIR" \
        --experiment-name "$EXPERIMENT_NAME"
    
    if [ $? -eq 0 ]; then
        print_success "Model training completed successfully"
        
        # Find best checkpoint
        CHECKPOINT_DIR="experiments/$EXPERIMENT_NAME/checkpoints"
        if [ -d "$CHECKPOINT_DIR" ]; then
            BEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -not -name "last.ckpt" | head -1)
            if [ -n "$BEST_CHECKPOINT" ]; then
                echo "Best checkpoint: $BEST_CHECKPOINT"
                export TRAINED_MODEL_CHECKPOINT="$BEST_CHECKPOINT"
            fi
        fi
    else
        print_error "Model training failed"
        exit 1
    fi
}

# =============================================================================
# EVALUATION
# =============================================================================

evaluate_model() {
    print_step "Evaluating Trained Model"
    
    if [ -z "$TRAINED_MODEL_CHECKPOINT" ]; then
        print_error "No trained model checkpoint found"
        return 1
    fi
    
    EVAL_OUTPUT_DIR="outputs/evaluation_$(date +%Y%m%d_%H%M%S)"
    
    echo "Checkpoint: $TRAINED_MODEL_CHECKPOINT"
    echo "Output directory: $EVAL_OUTPUT_DIR"
    echo ""
    
    python scripts/evaluate.py \
        --checkpoint "$TRAINED_MODEL_CHECKPOINT" \
        --output-dir "$EVAL_OUTPUT_DIR" \
        --splits test \
        --generate-plots \
        --num-samples 20
    
    if [ $? -eq 0 ]; then
        print_success "Model evaluation completed"
        echo "Evaluation results saved to: $EVAL_OUTPUT_DIR"
        
        # Show quick results if available
        if [ -f "$EVAL_OUTPUT_DIR/evaluation_report.json" ]; then
            echo ""
            echo "Quick results:"
            python -c "
import json
try:
    with open('$EVAL_OUTPUT_DIR/evaluation_report.json', 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics', {})
    print(f\"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\")
    print(f\"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}\")
    print(f\"  Lightning Detection Rate: {metrics.get('lightning_detection_rate', 'N/A'):.4f}\")
    print(f\"  False Alarm Ratio: {metrics.get('false_alarm_ratio', 'N/A'):.4f}\")
except:
    print('  Could not parse results')
"
        fi
    else
        print_error "Model evaluation failed"
        return 1
    fi
}

# =============================================================================
# MONITORING
# =============================================================================

setup_monitoring() {
    print_step "Setting Up Monitoring"
    
    # Check if tensorboard is available
    if command -v tensorboard &> /dev/null; then
        echo "Tensorboard is available. To monitor training, run:"
        echo "  tensorboard --logdir logs"
        echo ""
    else
        print_warning "Tensorboard not found - install with: pip install tensorboard"
    fi
    
    # Show log locations
    echo "Log locations:"
    echo "  Training logs: logs/"
    echo "  Experiment logs: experiments/$EXPERIMENT_NAME/logs/"
    echo "  Checkpoints: experiments/$EXPERIMENT_NAME/checkpoints/"
}

# =============================================================================
# CLEANUP AND SUMMARY
# =============================================================================

show_summary() {
    print_step "Pipeline Summary"
    
    echo "Completed steps:"
    echo "  ✓ Data preprocessing"
    echo "  ✓ Seasonal-aware data splits"
    echo "  ✓ Model training"
    echo "  ✓ Model evaluation"
    echo ""
    
    echo "Key outputs:"
    echo "  Processed data: $DATA_PROCESSED_DIR"
    echo "  Data splits: $DATA_SPLITS_DIR"
    echo "  Experiment: experiments/$EXPERIMENT_NAME"
    if [ -n "$TRAINED_MODEL_CHECKPOINT" ]; then
        echo "  Best model: $TRAINED_MODEL_CHECKPOINT"
    fi
    echo ""
    
    echo "Next steps:"
    echo "  1. Review evaluation results in outputs/"
    echo "  2. Test on 2022 data for final validation"
    echo "  3. Use scripts/inference.py for predictions on new data"
    echo "  4. Use scripts/transfer_domain.py for domain adaptation"
    echo ""
    
    echo "For more monitoring:"
    echo "  tensorboard --logdir logs"
}

# =============================================================================
# ERROR HANDLING
# =============================================================================

# Trap errors and cleanup
trap 'print_error "Pipeline failed at step: $BASH_COMMAND"; exit 1' ERR

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

# Help function
show_help() {
    echo "Lightning Prediction Model Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required (modify in script):"
    echo "  LIGHTNING_DIR     Directory with lightning NetCDF files"
    echo "  CAPE_FILE         CAPE NetCDF file path"
    echo "  TERRAIN_FILE      Terrain NetCDF file path"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  --skip-preprocess Skip data preprocessing step"
    echo "  --skip-training   Skip model training step"
    echo "  --skip-evaluation Skip model evaluation step"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run complete pipeline"
    echo "  $0 --skip-preprocess        # Skip preprocessing (data already processed)"
    echo "  $0 --skip-training          # Only preprocess data"
}

# Parse arguments
SKIP_PREPROCESS=false
SKIP_TRAINING=false
SKIP_EVALUATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

main() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "Lightning Prediction Model Pipeline"
    echo "=========================================="
    echo -e "${NC}"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run pipeline steps
    validate_inputs
    setup_directories
    setup_monitoring
    preprocess_data
    prepare_data_splits
    train_model
    evaluate_model
    
    # Calculate total time
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    SECONDS=$((TOTAL_TIME % 60))
    
    echo ""
    print_success "Pipeline completed successfully!"
    echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    
    show_summary
}

main_with_options() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "Lightning Prediction Model Pipeline"
    echo "=========================================="
    echo -e "${NC}"
    
    START_TIME=$(date +%s)
    
    # Always run validation and setup
    validate_inputs
    setup_directories
    setup_monitoring
    
    # Conditional steps
    if [ "$SKIP_PREPROCESS" = false ]; then
        preprocess_data
        prepare_data_splits
    else
        print_warning "Skipping data preprocessing"
    fi
    
    if [ "$SKIP_TRAINING" = false ]; then
        train_model
    else
        print_warning "Skipping model training"
        # Try to find existing checkpoint
        LATEST_EXPERIMENT=$(ls -t experiments/ 2>/dev/null | head -1)
        if [ -n "$LATEST_EXPERIMENT" ]; then
            CHECKPOINT_DIR="experiments/$LATEST_EXPERIMENT/checkpoints"
            if [ -d "$CHECKPOINT_DIR" ]; then
                TRAINED_MODEL_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -not -name "last.ckpt" | head -1)
                if [ -n "$TRAINED_MODEL_CHECKPOINT" ]; then
                    echo "Found existing checkpoint: $TRAINED_MODEL_CHECKPOINT"
                fi
            fi
        fi
    fi
    
    if [ "$SKIP_EVALUATION" = false ] && [ -n "$TRAINED_MODEL_CHECKPOINT" ]; then
        evaluate_model
    elif [ "$SKIP_EVALUATION" = false ]; then
        print_warning "Skipping evaluation - no trained model found"
    else
        print_warning "Skipping model evaluation"
    fi
    
    # Calculate total time
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    SECONDS=$((TOTAL_TIME % 60))
    
    echo ""
    print_success "Pipeline completed!"
    echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    
    show_summary
}

# Run main function with options
main_with_options