# SegFormer Tree Detector

An AI-powered tree detection system using SegFormer semantic segmentation model for satellite imagery analysis. This repository provides tools for training custom SegFormer models and running inference pipelines on GeoTIFF satellite images to detect and map tree coverage.

**Note:** This project was developed as part of my Bachelor's Thesis in Computer Science for academic research purposes. It is not intended for commercial use.

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

This project's code is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**DISCLAIMER:** This software is designed to work with `nvidia/mit-b3` model weights from Hugging Face Hub, which are subject to NVIDIA Source Code License (non-commercial use only). The use of this software with nvidia/mit-b3 weights is subject to their original licenses. **Users are responsible for complying with the licenses of the models they use.**

### License Breakdown:
- **This codebase**: Apache 2.0 (free for commercial and non-commercial use)
- **NVIDIA MIT-B3 weights**: NVIDIA Source Code License (non-commercial use only)
- **Hugging Face Transformers library**: Apache 2.0

### For Commercial Use:
✅ **You CAN**:
- Use this code commercially with your own trained models
- Use this code commercially with other SegFormer models that have permissive licenses
- Modify and distribute this codebase under Apache 2.0

❌ **You CANNOT**:
- Use NVIDIA's pre-trained MIT-B3 weights for commercial purposes
- Include NVIDIA's weights in commercial products without their permission

### Recommended Approach:
1. **For research/evaluation**: Use freely with NVIDIA weights
2. **For commercial use**:
   - Train your own model from scratch using this codebase, OR
   - Use alternative pre-trained models with commercial-friendly licenses

---

# SegFormer Tree Detector (Italiano)

Un sistema di rilevamento alberi basato su AI che utilizza il modello di segmentazione semantica SegFormer per l'analisi di immagini satellitari. Questo repository fornisce strumenti per addestrare modelli SegFormer personalizzati ed eseguire pipeline di inferenza su immagini satellitari GeoTIFF per rilevare e mappare la copertura arborea.

**Nota:** Questo progetto è stato sviluppato come parte della mia Tesi di Laurea Triennale in Informatica per scopi di ricerca accademica. Non è inteso per uso commerciale.

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

**Basato su:**
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Libreria ML all'avanguardia
- [SegFormer](https://arxiv.org/abs/2105.15203) - "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
- [MIT-B3 Model](https://huggingface.co/nvidia/mit-b3) - Pesi SegFormer pre-addestrati

**Caratteristiche Pipeline:**
- Elabora file singoli o intere directory
- Preserva i metadati geospaziali GeoTIFF
- Converte le maschere di rilevamento in coordinate poligonali
- Trasforma le coordinate pixel in coordinate geografiche
- Genera file GeoJSON per l'integrazione GIS
- Crea immagini overlay annotate

### Conversione Risultati di Rilevamento in GeoJSON

Converti file JSON di rilevamento in formato GeoJSON:

```bash
python json_to_geojson.py \
    --input ./results/json \
    --output ./results/detections.geojson
```

### Componenti Principali

#### `sf/train.py`
- **Scopo**: Addestra modelli SegFormer personalizzati sui tuoi dataset
- **Caratteristiche**: Caricamento automatico dati, preprocessing con processori HuggingFace, divisioni train/validation/test
- **Funzioni Chiave**:
  - `loadDataSimple()`: Carica immagini e maschere dalla struttura directory
  - `preprocessData()`: Applica trasformazioni HuggingFace e crea dataset
  - `trainSegformer()`: Pipeline di addestramento completa con parametri ottimizzati

#### `sf/inference.py`
- **Scopo**: Funzionalità di inferenza principale con classe SegFormerInference
- **Caratteristiche**: Operazioni tensor ottimizzate, caricamento modelli safetensors, visualizzazione
- **Classe Chiave**: `SegFormerInference`
  - Carica solo modelli safetensors (formato pronto per produzione)
  - Post-processing efficiente con interpolazione PyTorch
  - Soglia basata su confidenza per rilevamento alberi
  - Visualizzazione e statistiche integrate

#### `inference_pipeline.py`
- **Scopo**: Pipeline di produzione per elaborazione batch e generazione GeoJSON
- **Caratteristiche**: Supporto GeoTIFF, trasformazione coordinate, estrazione poligoni
- **Funzioni Chiave**:
  - `segformerDetector()`: Elabora singole immagini con consapevolezza geospaziale
  - `maskToPolygons()`: Converte maschere binarie in coordinate poligonali
  - `TreeDetectionPipeline`: Orchestrazione pipeline completa

#### `json_to_geojson.py`
- **Scopo**: Converte risultati di rilevamento in formati compatibili GIS
- **Caratteristiche**: Trasformazione poligoni, gestione CRS, unione multi-file
- **Funzioni Chiave**:
  - `segformerToGeoJsonFeature()`: Converte rilevamenti in feature GeoJSON
  - `mergeJsonToGeoJson()`: Combina più file di rilevamento

## Opzioni di Configurazione

### Parametri Inferenza
- `--confidence`: Soglia di confidenza rilevamento (0.0-1.0, default: 0.5)
- `--batch`: Abilita elaborazione batch di directory
- `--no_visualize`: Disabilita output visualizzazione
- `--output_dir`: Specifica directory output

### Parametri Pipeline
- `--no-images`: Salta salvataggio immagini overlay annotate
- `--no-json`: Salta salvataggio file JSON rilevamento
- `--no-geojson`: Salta conversione GeoJSON
- `--verbose`: Abilita logging dettagliato

### Parametri Addestramento
- `--epochs`: Numero di epoche di addestramento
- `--learning_rate`: Tasso di apprendimento ottimizzatore
- `--batch_size`: Dimensione batch addestramento
- Ottimizzazione GPU: Precisione mista automatica, ottimizzatori fused, caricamento dati parallelo

## Formati Output

### Output Inferenza Diretta
- **Maschere Binarie**: `{basename}_ultra_mask.png` - Maschere binarie albero/non-albero
- **Maschere Colorate**: `{basename}_ultra_colored_mask.png` - Segmentazione visualizzata
- **Grafici Analisi**: `{basename}_ultra_sensitive_results.png` - Heatmap probabilità e statistiche

### Output Pipeline
```
output/
├── annotated/
│   ├── image1_segformer_annotated.tif    # Immagini overlay
│   └── image2_segformer_annotated.tif
├── json/
│   ├── image1_results.json               # Dati rilevamento
│   ├── image2_results.json
│   ├── pipeline_summary.json             # Statistiche elaborazione
│   └── detections.geojson               # GeoJSON combinato
```

### Struttura Dati Rilevamento
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

## Utilizzo Avanzato

### Integrazione Modello Personalizzato
```python
from sf.inference import SegFormerInference

# Inizializza con modello personalizzato
segformer = SegFormerInference(
    modelPath="./my_model.safetensors",
    numClasses=2,
    confidence=0.7
)

# Elabora immagine
mask, colored_mask = segformer.predict(
    imagePath="satellite_image.tif",
    outputDir="./results",
    visualize=True
)
```

### Elaborazione Batch con Logica Personalizzata
```python
from inference_pipeline import TreeDetectionPipeline

# Inizializza pipeline
pipeline = TreeDetectionPipeline(
    modelType="segformer",
    modelPath="./model.safetensors",
    confidence=0.6
)

# Esegui pipeline completa
results = pipeline.runPipeline(
    inputPath="/path/to/geotiff/folder",
    outputFolder="./results",
    saveImages=True,
    saveJson=True,
    createGeojson=True
)
```

## Risoluzione Problemi

### Problemi Comuni

**Problemi Memoria GPU**
- Riduci batch size nell'addestramento: `--batch_size 1`
- Usa accumulo gradiente (configurato automaticamente)
- Elabora tile immagine più piccoli

**Errori Caricamento Modello**
- Assicurati che il percorso modello punti al file `.safetensors`
- Verifica che il modello sia stato addestrato con versione SegFormer compatibile
- Controlla compatibilità CUDA per inferenza GPU

**Problemi Elaborazione GeoTIFF**
- Verifica installazione rasterio: `pip install rasterio`
- Controlla permessi file e percorsi
- Assicurati che i file GeoTIFF abbiano informazioni CRS valide

### Ottimizzazione Prestazioni

**Velocità Addestramento**
- Usa GPU compatibile CUDA
- Aumenta batch size se la memoria lo permette
- Abilita addestramento precisione mista (automatico)
- Usa ottimizzatori fused (automatico su GPU)

**Velocità Inferenza**
- Usa formato modello safetensors (default)
- Elabora più immagini in batch
- Disabilita visualizzazione per produzione: `--no_visualize`
- Usa soglie di confidenza appropriate

## Licenza

Il codice di questo progetto è licenziato sotto la Apache License 2.0 - vedi il file [LICENSE](LICENSE) per i dettagli.

**DISCLAIMER:** Questo software è progettato per funzionare con i pesi del modello `nvidia/mit-b3` da Hugging Face Hub, che sono soggetti alla NVIDIA Source Code License (solo uso non commerciale). L'utilizzo di questo software con i pesi nvidia/mit-b3 è soggetto alle loro licenze originali. **L'utente è responsabile del rispetto delle licenze dei modelli utilizzati.**

### Dettaglio Licenze:
- **Questo codebase**: Apache 2.0 (libero per uso commerciale e non commerciale)
- **Pesi NVIDIA MIT-B3**: NVIDIA Source Code License (solo uso non commerciale)
- **Libreria Hugging Face Transformers**: Apache 2.0

### Per Uso Commerciale:
✅ **PUOI**:
- Usare questo codice commercialmente con i tuoi modelli addestrati
- Usare questo codice commercialmente con altri modelli SegFormer con licenze permissive
- Modificare e distribuire questo codebase sotto Apache 2.0

❌ **NON PUOI**:
- Usare i pesi pre-addestrati MIT-B3 di NVIDIA per scopi commerciali
- Includere i pesi NVIDIA in prodotti commerciali senza il loro permesso

### Approccio Consigliato:
1. **Per ricerca/valutazione**: Usa liberamente con i pesi NVIDIA
2. **Per uso commerciale**:
   - Addestra il tuo modello da zero usando questo codebase, OPPURE
   - Usa modelli pre-addestrati alternativi con licenze commerciali permissive