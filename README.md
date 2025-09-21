# SegFormer Tree Detector

An AI-powered tree detection system using SegFormer semantic segmentation model for satellite imagery analysis. This repository provides tools for training custom SegFormer models and running inference pipelines on GeoTIFF satellite images to detect and map tree coverage.

## Project Overview

This project leverages [Hugging Face's SegFormer implementation](https://github.com/huggingface/transformers) to perform semantic segmentation for tree detection in satellite imagery. The system uses the [`nvidia/mit-b3`](https://huggingface.co/nvidia/mit-b3) model from Hugging Face Hub and can process both single images and batch folders of GeoTIFF tiles, producing detailed tree coverage maps with polygon extraction and GeoJSON output for GIS integration.

**Based on:**
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - State-of-the-art ML library
- [SegFormer](https://arxiv.org/abs/2105.15203) - "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
- [MIT-B3 Model](https://huggingface.co/nvidia/mit-b3) - Pre-trained SegFormer weights

### Key Features

- **Custom SegFormer Training**: Train models on your own datasets with automatic data preprocessing
- **GeoTIFF Support**: Full geospatial metadata preservation and coordinate transformation
- **Batch Processing**: Process entire folders of satellite tiles efficiently
- **Multiple Output Formats**: Binary masks, colored overlays, polygon coordinates, and GeoJSON
- **Production Ready**: Optimized inference with safetensors model format
- **Easy Integration**: Simple Python API and command-line interface

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Dependencies
```
torch>=2.4.0
torchvision>=0.17.0
transformers>=4.35.0
datasets>=2.14.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
rasterio>=1.4.0
safetensors>=0.4.0
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/segformer_tree_detector.git
cd segformer_tree_detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python sf/inference.py --help
```

## Dataset Structure

For training, organize your dataset as follows:
```
dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── masks/
    ├── image_001_mask.png
    ├── image_002_mask.png
    └── ...
```

**Mask Requirements:**
- Binary masks: 0 for background, 255 for trees
- Same base filename as corresponding image
- Supported formats: PNG, JPG
- Naming patterns: `{basename}_mask.{ext}` or `{basename}.{ext}`

## Usage

### Training a Custom Model

Train a SegFormer model on your dataset:

```bash
python sf/train.py \
    --data_dir /path/to/dataset \
    --output_dir ./trained_model \
    --epochs 40 \
    --learning_rate 0.00006 \
    --batch_size 2
```

**Training Parameters:**
- `--data_dir`: Directory containing images and masks folders
- `--output_dir`: Where to save the trained model (default: ./segformer_results)
- `--epochs`: Number of training epochs (default: 40)
- `--learning_rate`: Learning rate (default: 0.00006)
- `--batch_size`: Batch size (default: 2)

The training script automatically:
- Loads and preprocesses images using HuggingFace processors
- Splits data into train/validation/test sets (70/20/10)
- Applies data augmentation and normalization
- Saves the best model as safetensors format
- Uses mixed precision training for speed optimization

### Single Image Inference

Process a single image:

```bash
python sf/inference.py \
    --model_path ./trained_model/model.safetensors \
    --image_path satellite_image.tif \
    --output_dir ./results \
    --confidence 0.5
```

### Batch Processing

Process multiple images:

```bash
python sf/inference.py \
    --model_path ./trained_model/model.safetensors \
    --image_path /path/to/images/ \
    --output_dir ./results \
    --batch \
    --confidence 0.5
```

### Complete Pipeline with GeoJSON Output

Run the full detection pipeline:

```bash
python inference_pipeline.py \
    --input /path/to/geotiff/files \
    --model-path ./trained_model/model.safetensors \
    --output ./pipeline_results \
    --confidence 0.5
```

**Pipeline Features:**
- Processes single files or entire directories
- Preserves GeoTIFF geospatial metadata
- Converts detection masks to polygon coordinates
- Transforms pixel coordinates to geographic coordinates
- Generates GeoJSON files for GIS integration
- Creates annotated overlay images

### Convert Detection Results to GeoJSON

Convert JSON detection files to GeoJSON format:

```bash
python json_to_geojson.py \
    --input ./results/json \
    --output ./results/detections.geojson

### Core Components

#### `sf/train.py`
- **Purpose**: Train custom SegFormer models on your datasets
- **Features**: Automatic data loading, preprocessing with HuggingFace processors, train/validation/test splits
- **Key Functions**:
  - `loadDataSimple()`: Load images and masks from directory structure
  - `preprocessData()`: Apply HuggingFace transformations and create datasets
  - `trainSegformer()`: Complete training pipeline with optimized parameters

#### `sf/inference.py`
- **Purpose**: Core inference functionality with SegFormerInference class
- **Features**: Optimized tensor operations, safetensors model loading, visualization
- **Key Class**: `SegFormerInference`
  - Loads safetensors models only (production-ready format)
  - Efficient post-processing with PyTorch interpolation
  - Confidence-based thresholding for tree detection
  - Built-in visualization and statistics

#### `inference_pipeline.py`
- **Purpose**: Production pipeline for batch processing and GeoJSON generation
- **Features**: GeoTIFF support, coordinate transformation, polygon extraction
- **Key Functions**:
  - `segformerDetector()`: Process single images with geospatial awareness
  - `maskToPolygons()`: Convert binary masks to polygon coordinates
  - `TreeDetectionPipeline`: Complete pipeline orchestration

#### `json_to_geojson.py`
- **Purpose**: Convert detection results to GIS-compatible formats
- **Features**: Polygon transformation, CRS handling, multi-file merging
- **Key Functions**:
  - `segformerToGeoJsonFeature()`: Convert detections to GeoJSON features
  - `mergeJsonToGeoJson()`: Combine multiple detection files

## Configuration Options

### Inference Parameters
- `--confidence`: Detection confidence threshold (0.0-1.0, default: 0.5)
- `--batch`: Enable batch processing of directories
- `--no_visualize`: Disable visualization output
- `--output_dir`: Specify output directory

### Pipeline Parameters
- `--no-images`: Skip saving annotated overlay images
- `--no-json`: Skip saving JSON detection files
- `--no-geojson`: Skip GeoJSON conversion
- `--verbose`: Enable detailed logging

### Training Parameters
- `--epochs`: Number of training epochs
- `--learning_rate`: Optimizer learning rate
- `--batch_size`: Training batch size
- GPU optimization: Automatic mixed precision, fused optimizers, parallel data loading

## Output Formats

### Direct Inference Outputs
- **Binary Masks**: `{basename}_ultra_mask.png` - Binary tree/no-tree masks
- **Colored Masks**: `{basename}_ultra_colored_mask.png` - Visualized segmentation
- **Analysis Plots**: `{basename}_ultra_sensitive_results.png` - Probability heatmaps and statistics

### Pipeline Outputs
```
output/
├── annotated/
│   ├── image1_segformer_annotated.tif    # Overlay images
│   └── image2_segformer_annotated.tif
├── json/
│   ├── image1_results.json               # Detection data
│   ├── image2_results.json
│   ├── pipeline_summary.json             # Processing statistics
│   └── detections.geojson               # Combined GeoJSON
```

### Detection Data Structure
```json
{
  "image_path": "path/to/image.tif",
  "model_type": "segformer",
  "confidence_threshold": 0.5,
  "trees_detected": 25,
  "tree_coverage_percent": 12.5,
  "polygons": [[[x1,y1], [x2,y2], ...]],
  "geo_polygons": [{"polygon": [[lon1,lat1], [lon2,lat2], ...]}],
  "crs": "EPSG:4326",
  "transform": [...]
}
```

## Advanced Usage

### Custom Model Integration
```python
from sf.inference import SegFormerInference

# Initialize with custom model
segformer = SegFormerInference(
    modelPath="./my_model.safetensors",
    numClasses=2,
    confidence=0.7
)

# Process image
mask, colored_mask = segformer.predict(
    imagePath="satellite_image.tif",
    outputDir="./results",
    visualize=True
)
```

### Batch Processing with Custom Logic
```python
from inference_pipeline import TreeDetectionPipeline

# Initialize pipeline
pipeline = TreeDetectionPipeline(
    modelType="segformer",
    modelPath="./model.safetensors",
    confidence=0.6
)

# Run complete pipeline
results = pipeline.runPipeline(
    inputPath="/path/to/geotiff/folder",
    outputFolder="./results",
    saveImages=True,
    saveJson=True,
    createGeojson=True
)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**License Notes:**
- This project uses [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache 2.0 license)
- SegFormer model weights are used through [Hugging Face Hub](https://huggingface.co/nvidia/mit-b3) (publicly available)
- Your custom training and inference code is Apache 2.0 licensed
- Free for commercial and non-commercial use
- Must include license and copyright notice when redistributing

---

# SegFormer Tree Detector (Italiano)

Un sistema di rilevamento alberi basato su AI che utilizza il modello di segmentazione semantica SegFormer per l'analisi di immagini satellitari. Questo repository fornisce strumenti per addestrare modelli SegFormer personalizzati ed eseguire pipeline di inferenza su immagini satellitari GeoTIFF per rilevare e mappare la copertura arborea.

## Panoramica del Progetto

Questo progetto sfrutta l'implementazione SegFormer di Hugging Face per eseguire segmentazione semantica per il rilevamento di alberi in immagini satellitari. Il sistema può elaborare sia immagini singole che cartelle batch di tile GeoTIFF, producendo mappe dettagliate della copertura arborea con estrazione di poligoni e output GeoJSON per l'integrazione GIS.

### Caratteristiche Principali

- **Addestramento SegFormer Personalizzato**: Addestra modelli sui tuoi dataset con preprocessing automatico dei dati
- **Supporto GeoTIFF**: Preservazione completa dei metadati geospaziali e trasformazione delle coordinate
- **Elaborazione Batch**: Elabora efficientemente intere cartelle di tile satellitari
- **Formati di Output Multipli**: Maschere binarie, overlay colorati, coordinate poligonali e GeoJSON
- **Pronto per Produzione**: Inferenza ottimizzata con formato modello safetensors
- **Integrazione Semplice**: API Python semplice e interfaccia a riga di comando

## Requisiti

### Requisiti di Sistema
- Python 3.8+
- GPU compatibile CUDA (raccomandato per l'addestramento)
- 8GB+ RAM (16GB+ raccomandati per dataset grandi)

### Dipendenze
```
torch>=2.4.0
torchvision>=0.17.0
transformers>=4.35.0
datasets>=2.14.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
rasterio>=1.4.0
safetensors>=0.4.0
```

## Installazione

1. **Clona il repository**
```bash
git clone https://github.com/your-username/segformer_tree_detector.git
cd segformer_tree_detector
```

2. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

3. **Verifica l'installazione**
```bash
python sf/inference.py --help
```

## Struttura Dataset

Per l'addestramento, organizza il tuo dataset come segue:
```
dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── masks/
    ├── image_001_mask.png
    ├── image_002_mask.png
    └── ...
```

**Requisiti Maschere:**
- Maschere binarie: 0 per sfondo, 255 per alberi
- Stesso nome base dell'immagine corrispondente
- Formati supportati: PNG, JPG
- Pattern di denominazione: `{basename}_mask.{ext}` o `{basename}.{ext}`

## Utilizzo

### Addestramento di un Modello Personalizzato

Addestra un modello SegFormer sul tuo dataset:

```bash
python sf/train.py \
    --data_dir /path/to/dataset \
    --output_dir ./trained_model \
    --epochs 40 \
    --learning_rate 0.00006 \
    --batch_size 2
```

### Inferenza Immagine Singola

Elabora una singola immagine:

```bash
python sf/inference.py \
    --model_path ./trained_model/model.safetensors \
    --image_path satellite_image.tif \
    --output_dir ./results \
    --confidence 0.5
```

### Elaborazione Batch

Elabora multiple immagini:

```bash
python sf/inference.py \
    --model_path ./trained_model/model.safetensors \
    --image_path /path/to/images/ \
    --output_dir ./results \
    --batch \
    --confidence 0.5
```

### Pipeline Completa con Output GeoJSON

Esegui la pipeline di rilevamento completa:

```bash
python inference_pipeline.py \
    --input /path/to/geotiff/files \
    --model-path ./trained_model/model.safetensors \
    --output ./pipeline_results \
    --confidence 0.5
```

## Licenza

Questo progetto è licenziato sotto la Apache License 2.0 - vedi il file [LICENSE](LICENSE) per i dettagli.

**Note sulla Licenza:**
- Questo progetto utilizza [Hugging Face Transformers](https://github.com/huggingface/transformers) (licenza Apache 2.0)
- I pesi del modello SegFormer sono utilizzati tramite [Hugging Face Hub](https://huggingface.co/nvidia/mit-b3) (pubblicamente disponibili)
- Il tuo codice personalizzato di training e inferenza è sotto licenza Apache 2.0
- Libero per uso commerciale e non commerciale
- Deve includere avviso di licenza e copyright quando ridistribuito