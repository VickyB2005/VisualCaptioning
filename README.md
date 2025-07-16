# Visual Captioning with Scene Context Integration

A Deep Learning Approach Using MSVD Dataset for generating natural language descriptions of short action videos through an encoder-decoder architecture with attention mechanisms and scene context integration.

## Overview

This project implements a sophisticated video captioning system that combines computer vision and natural language processing to generate descriptive captions for short action videos. The system uses a CNN-LSTM encoder-decoder architecture enhanced with attention mechanisms and scene context integration to produce semantically accurate and contextually relevant captions.

## Key Features

- **CNN-LSTM Architecture**: VGG16 for spatial feature extraction and LSTM for temporal modeling
- **Attention Mechanism**: Dynamic attention to focus on relevant visual features during caption generation
- **Scene Context Integration**: Environmental cues using Places365-inspired scene classification
- **Dual Decoding Strategies**: Both greedy search and beam search for caption generation
- **Real-time Prediction**: Live caption generation for video inputs
- **Comprehensive Evaluation**: Multiple metrics including BLEU, METEOR, and CIDEr scores

## Sample Results

The system generates contextually aware captions for various video types:

| Video Type | Beam Search Caption | Greedy Search Caption | Time Comparison |
|------------|--------------------|-----------------------|-----------------|
| Cooking Video | "a woman is seasoning some food" | "a woman is seasoning some food" | 22.05s vs 0.70s |
| Performance | "a man is singing" | "a man is performing on a stage" | 13.79s vs 0.77s |
| Sports Activity | "a man is riding a bicycle" | "a man is riding a bicycle" | 22.20s vs 0.66s |
| Pet Activity | "a cat is playing the piano" | "a cat is playing the piano" | 26.48s vs 0.70s |

## Dataset

The project uses the **MSVD (Microsoft Research Video Description) dataset**:
- **Training Videos**: 1,200 videos with multiple human-annotated captions
- **Validation Videos**: 100 videos for hyperparameter tuning
- **Test Videos**: 670 videos for final evaluation
- **Average Duration**: 10.2 seconds per clip
- **Captions**: ~40 natural language descriptions per video

## Architecture

### System Components

1. **Visual Feature Extraction Module**
   - Uniform temporal sampling (80 frames per video)
   - VGG16 CNN for spatial feature extraction
   - 4096-dimensional feature vectors per frame

2. **Temporal Encoding Module**
   - LSTM encoder with 512 hidden units
   - Captures temporal dependencies and action dynamics
   - Processes sequential visual information

3. **Attention-Based Decoder**
   - LSTM decoder with attention mechanism
   - Dynamic focus on relevant visual features
   - Vocabulary size: 1,500 words
   - Maximum caption length: 10 words

4. **Scene Context Integration**
   - Places365-inspired scene classification
   - Environmental context embedding
   - Enhanced contextual understanding

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (recommended)
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Visual-Captioning.git
   cd Visual-Captioning
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv310
   
   # On Windows
   venv310\Scripts\activate
   
   # On macOS/Linux
   source venv310/bin/activate
   ```

3. **Upgrade pip and install dependencies**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install specific versions (if needed)**
   ```bash
   pip install tensorflow==2.10 keras==2.10
   pip install "numpy<2.0"
   pip install tqdm opencv-python scipy scikit-learn h5py matplotlib pillow
   ```

## Usage

### Quick Start - Prediction

1. **Real-time prediction on test videos**
   ```bash
   python predict_realtime.py
   ```

2. **Batch prediction with performance metrics**
   ```bash
   python predict_test.py
   ```

3. **Extract features from new videos**
   ```bash
   python extract_features.py
   ```

### Training from Scratch

1. **Prepare your dataset**
   - Place training videos in `data/training_data/video/`
   - Place test videos in `data/testing_data/video/`

2. **Extract features**
   ```bash
   python extract_features.py
   ```

3. **Train the model**
   ```bash
   python train.py
   ```

4. **Monitor training with Jupyter notebook**
   ```bash
   jupyter notebook Video_Captioning.ipynb
   ```

## Model Architecture Details

### Training Architecture
- **Encoder**: VGG16 + LSTM (512 units)
- **Decoder**: LSTM (512 units) + Attention
- **Batch Size**: 320
- **Learning Rate**: 0.0007 with Adam optimizer
- **Training Epochs**: 150 with early stopping
- **Regularization**: Dropout (0.5) and gradient clipping

### Performance Metrics
- **Training Accuracy**: ~80%
- **Validation Accuracy**: ~74%
- **BLEU-4 Score**: 0.35 (Beam Search)
- **METEOR Score**: 0.27
- **CIDEr Score**: 0.39

## Project Structure

```
Visual-Captioning/
├── data/
│   ├── training_data/
│   │   ├── video/          # Training videos
│   │   └── feat/           # Extracted features
│   └── testing_data/
│       ├── video/          # Test videos
│       └── feat/           # Test features
├── model_final/
│   ├── encoder_model.h5    # Trained encoder
│   ├── decoder_model_weights.h5  # Decoder weights
│   └── tokenizer1500      # Vocabulary tokenizer
├── images/                 # Result visualizations
├── config.py              # Configuration parameters
├── extract_features.py    # Feature extraction script
├── train.py              # Training script
├── predict_realtime.py   # Real-time prediction
├── predict_test.py       # Batch prediction
├── scene_classifier.py   # Scene context module
└── requirements.txt      # Dependencies
```

## Decoding Strategies

### Greedy Search
- **Speed**: ~0.70 seconds per video
- **Accuracy**: 72%
- **Best for**: Real-time applications

### Beam Search (Beam Width: 3)
- **Speed**: ~22 seconds per video
- **Accuracy**: 74%
- **Best for**: High-quality caption generation

## Key Scripts

- **`train.py`**: Complete model training pipeline
- **`predict_realtime.py`**: Real-time video captioning
- **`predict_test.py`**: Batch prediction with performance metrics
- **`extract_features.py`**: Feature extraction from videos
- **`scene_classifier.py`**: Scene context classification
- **`config.py`**: Configuration and hyperparameters

## Performance Comparison

| Model | Year | BLEU-4 | Architecture |
|-------|------|--------|--------------|
| LRCN | 2015 | - | C3D + LSTM |
| S2VT | 2015 | 28.1 | Seq2Seq + Attention |
| HRNE | 2016 | 29.2 | Hierarchical RNN |
| SCN-LSTM | 2017 | 30.5 | Semantic Composition |
| **Our Model** | 2025 | **35.0** | **CNN-LSTM + Attention + Scene Context** |

## Future Enhancements

- Integration of transformer-based architectures
- Support for longer video sequences
- Advanced attention mechanisms (multi-head attention)
- Real-time web interface
- Multi-language caption generation
- Integration with modern video understanding models (I3D, Video Transformer)

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size in `config.py`
   - Use CPU-only mode for testing

2. **Dependency Conflicts**
   ```bash
   pip uninstall keras tensorflow --yes
   pip install keras==2.10 tensorflow==2.10
   ```

3. **Video Loading Issues**
   - Ensure OpenCV is properly installed
   - Check video file formats (MP4 recommended)

## Technical Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: 5GB for dataset and models
- **Python**: 3.7-3.10

## Model Files

Pre-trained models are available in the `model_final/` directory:
- `encoder_model.h5`: Complete encoder architecture
- `decoder_model_weights.h5`: Decoder weights
- `tokenizer1500`: Vocabulary tokenizer

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

