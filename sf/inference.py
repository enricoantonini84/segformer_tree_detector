#Segformer inference script with Hugging Face transformers libraries
# this script is also used by inference_pipeline to predict trees
# in multiple images

import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
import glob
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor
)
from huggingface_hub import hf_hub_download
from transformers.utils import CONFIG_NAME
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import shutil
import time

# set environment variables to help with HuggingFace caching and rate limiting
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

# Additional caching directory for model configurations
MODEL_CACHE_DIR = os.path.expanduser('~/.cache/segformer_models')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using official Hugging Face SegFormer implementation with enhanced caching
class SegFormerInference:
    # Class-level cache for processor and base model to avoid multiple downloads
    _processor_cache = None
    _base_model_cache = None
    _last_cache_check = 0
    _cache_check_interval = 300  # 5 minutes
    
    @classmethod
    def _ensure_model_cached(cls, model_name="nvidia/mit-b3"):
        """Ensure the model configuration and processor are cached locally."""
        current_time = time.time()
        
        # Only check cache periodically to avoid repeated filesystem operations
        if current_time - cls._last_cache_check < cls._cache_check_interval and cls._processor_cache is not None:
            return
        
        cls._last_cache_check = current_time
        config_cache_path = os.path.join(MODEL_CACHE_DIR, f"{model_name.replace('/', '_')}_config.json")
        
        # Download and cache config if not exists
        if not os.path.exists(config_cache_path):
            try:
                print(f"Downloading and caching config for {model_name}...")
                downloaded_config = hf_hub_download(
                    repo_id=model_name,
                    filename=CONFIG_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    local_files_only=False
                )
                shutil.copy2(downloaded_config, config_cache_path)
                print(f"Config cached to {config_cache_path}")
            except Exception as e:
                print(f"Warning: Could not cache config: {e}")
        
        # Load processor with enhanced caching
        if cls._processor_cache is None:
            print("Loading SegFormer processor (one-time setup)...")
            try:
                cls._processor_cache = SegformerImageProcessor.from_pretrained(
                    model_name,
                    cache_dir=os.environ['TRANSFORMERS_CACHE'],
                    local_files_only=False
                )
                print("Processor loaded and cached successfully")
            except Exception as e:
                print(f"Error loading processor: {e}")
                # Retry with longer timeout
                time.sleep(2)
                cls._processor_cache = SegformerImageProcessor.from_pretrained(model_name)
    
    def __init__(self, modelPath: str, numClasses: int = 2, inputSize: Tuple[int, int] = (512, 512),
                 confidence: float = 0.5):
        self.modelPath = modelPath
        self.numClasses = numClasses
        self.inputSize = inputSize
        self.confidence = confidence
        self.device = device
        
        # Ensure model is cached before proceeding
        self._ensure_model_cached()
        
        # Use cached processor
        self.imageProcessor = self._processor_cache
        
        # load model (this will handle the model loading once)
        self.model = self.loadModel()
        
        # define color mapping for tree cover
        self.colorMap = np.array([
            [0, 0, 0],      # Background - black
            [0, 255, 0],    # Tree cover - green
        ])

    #load safetensor file model trained by train.py
    def loadModel(self) -> nn.Module:
        print(f"Loading model from {self.modelPath}")
        
        if not os.path.exists(self.modelPath) or not self.modelPath.endswith('.safetensors'):
            raise ValueError(f"Model path must be a .safetensors file. Got: {self.modelPath}")
        
        # Use cached base model if available
        if self._base_model_cache is None:
            print("Loading base SegFormer model with enhanced caching...")
            try:
                # Try loading from cache first (offline mode)
                self._base_model_cache = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/mit-b3",
                    num_labels=self.numClasses,
                    ignore_mismatched_sizes=True,
                    cache_dir=os.environ['TRANSFORMERS_CACHE'],
                    local_files_only=True
                )
                print("Base model loaded from cache successfully")
            except Exception as cache_e:
                print(f"Cache miss for base model: {cache_e}")
                print("Downloading base model...")
                
                # Temporarily enable online mode for download
                old_offline = os.environ.get('HF_HUB_OFFLINE', '0')
                old_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', '0')
                os.environ['HF_HUB_OFFLINE'] = '0'
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
                try:
                    self._base_model_cache = SegformerForSemanticSegmentation.from_pretrained(
                        "nvidia/mit-b3",
                        num_labels=self.numClasses,
                        ignore_mismatched_sizes=True,
                        cache_dir=os.environ['TRANSFORMERS_CACHE'],
                        local_files_only=False
                    )
                    print("Base model downloaded and cached successfully")
                finally:
                    # Restore offline mode
                    os.environ['HF_HUB_OFFLINE'] = old_offline
                    os.environ['TRANSFORMERS_OFFLINE'] = old_transformers_offline
        
        # Create a new instance from the cached model
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b3",
                num_labels=self.numClasses,
                ignore_mismatched_sizes=True,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                local_files_only=True  # Force offline loading
            )
        except Exception as e:
            print(f"Error creating model instance from cache: {e}")
            # Use the cached instance directly if we can't create a new one
            model = self._base_model_cache
        
        # Load fine-tuned weights
        print(f"Loading fine-tuned weights from {self.modelPath}")
        from safetensors.torch import load_file
        state_dict = load_file(self.modelPath)
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded fine-tuned weights")
        
        model.to(self.device)
        model.eval()
        return model
    
    # postprocess model prediction using confidence threshold
    def postprocessPrediction(self, prediction: torch.Tensor, target_size: tuple) -> tuple:
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(prediction, dim=1)
        
        # For binary classification, use confidence threshold
        if self.numClasses == 2:
            # Get tree probability map at original resolution using efficient interpolation
            treeProb = probabilities[0, 1]
            treeProbResized = torch.nn.functional.interpolate(
                treeProb.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # Debug output to understand what's happening
            print(f"Tree probability stats: min={treeProbResized.min():.6f}, max={treeProbResized.max():.6f}, mean={treeProbResized.mean():.6f}")
            print(f"Using confidence threshold: {self.confidence}")
            
            mask = (treeProbResized > self.confidence).astype(np.uint8)
            treePixels = mask.sum()
            print(f"Threshold result: {treePixels:,} tree pixels ({(treePixels/mask.size)*100:.2f}%)")
            
            # If too many pixels are classified as trees, there might be an issue
            if (treePixels/mask.size) > 0.8:
                print("WARNING: More than 80% of pixels classified as trees - this might indicate a problem")
                print("Tree probability distribution:")
                probHist, bins = np.histogram(treeProbResized, bins=10)
                for i, (count, binEdge) in enumerate(zip(probHist, bins[:-1])):
                    print(f"  {binEdge:.3f}-{bins[i+1]:.3f}: {count} pixels")
            
            return mask, treeProbResized
        else:
            # Multi-class fallback to argmax
            predictionArgmax = torch.argmax(probabilities, dim=1)
            # Resize to target size
            mask = torch.nn.functional.interpolate(
                predictionArgmax.float().unsqueeze(1),
                size=target_size,
                mode='nearest'
            ).squeeze().cpu().numpy().astype(np.uint8)
            return mask, None
    
    def predict(self, imagePath: str, outputDir: Optional[str] = None,
                visualize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Perform ultra-sensitive inference on a single image"""
        # load image
        print(f"Processing image: {imagePath}")
        image = cv2.imread(imagePath)
        
        if image is None:
            raise ValueError(f"Could not load image from {imagePath}")
        
        originalSize = image.shape[:2]
        # preprocess using HuggingFace processor
        imagePil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.imageProcessor(imagePil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits
        
        # store prediction for visualization
        self.lastPrediction = prediction.clone()
        self.lastProbabilities = None
        
        # postprocess using HuggingFace optimized utilities
        mask, probabilities = self.postprocessPrediction(prediction, originalSize)
        if probabilities is not None:
            self.lastProbabilities = probabilities
        
        # create colored mask
        coloredMask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for classId in range(self.numClasses):
            coloredMask[mask == classId] = self.colorMap[classId]
        
        # calculate and report statistics
        treePixels = (mask == 1).sum()
        totalPixels = mask.size
        treeCoverage = (treePixels / totalPixels) * 100
        print(f"Tree coverage: {treeCoverage:.2f}% ({treePixels:,}/{totalPixels:,} pixels)")
        
        if visualize:
            self.visualizeResults(image, mask, coloredMask, imagePath, outputDir)
        
        if outputDir:
            self.saveResults(mask, coloredMask, imagePath, outputDir)
        
        return mask, coloredMask
    
    #results visualization
    def visualizeResults(self, image: np.ndarray, mask: np.ndarray,
                        coloredMask: np.ndarray, imagePath: str,
                        outputDir: Optional[str] = None):
        """Visualize results with enhanced probability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Overlay
        imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(imageRgb, 0.4, coloredMask, 0.6, 0)
        axes[0,1].imshow(overlay)
        treeCount = (mask == 1).sum()
        treeCoverage = (treeCount / mask.size) * 100
        axes[0,1].set_title(f'Tree Detection Overlay\n({treeCount:,} pixels, {treeCoverage:.2f}%)')
        axes[0,1].axis('off')
        
        if hasattr(self, 'lastProbabilities') and self.lastProbabilities is not None:
            # Use the already processed probabilities at correct resolution
            treeProb = self.lastProbabilities
            
            # Enhance contrast for visualization
            probMin, probMax = treeProb.min(), treeProb.max()
            treeProbEnhanced = (treeProb - probMin) / (probMax - probMin) if probMax > probMin else treeProb
            
            im1 = axes[1,0].imshow(treeProbEnhanced, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1,0].set_title(f'Tree Probability Heatmap\n(Range: {probMin:.6f}-{probMax:.6f})')
            axes[1,0].axis('off')
            plt.colorbar(im1, ax=axes[1,0], shrink=0.8)
            
            # Probability histogram
            axes[1,1].hist(treeProb.flatten(), bins=50, alpha=0.7, color='green')
            axes[1,1].set_title('Tree Probability Distribution')
            axes[1,1].set_xlabel('Probability')
            axes[1,1].set_ylabel('Pixel Count')
            axes[1,1].axvline(treeProb.mean(), color='red', linestyle='--', label=f'Mean: {treeProb.mean():.6f}')
            axes[1,1].axvline(np.percentile(treeProb, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(treeProb, 95):.6f}')
            axes[1,1].legend()
        else:
            # If no prediction data, show empty plots
            axes[1,0].axis('off')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)
            baseName = os.path.splitext(os.path.basename(imagePath))[0]
            plt.savefig(os.path.join(outputDir, f"{baseName}_ultra_sensitive_results.png"),
                       dpi=150, bbox_inches='tight')
            print(f"Ultra-sensitive visualization saved to {outputDir}")
        
        plt.show()
    
    def saveResults(self, mask: np.ndarray, coloredMask: np.ndarray,
                   imagePath: str, outputDir: str):
        """Save inference results to files"""
        os.makedirs(outputDir, exist_ok=True)
        baseName = os.path.splitext(os.path.basename(imagePath))[0]
        
        # Save binary mask
        cv2.imwrite(os.path.join(outputDir, f"{baseName}_ultra_mask.png"), mask * 255)
        
        # Save colored mask
        cv2.imwrite(os.path.join(outputDir, f"{baseName}_ultra_colored_mask.png"),
                   cv2.cvtColor(coloredMask, cv2.COLOR_RGB2BGR))
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SegFormer Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained SegFormer model")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="Output directory")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--batch", action="store_true",
                       help="Batch process all images in image_path directory")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Skip visualization")
    
    args = parser.parse_args()
    
    print("Starting SegFormer inference!")
    
    try:
        # Initialize SegFormerInference for inference
        segformer = SegFormerInference(
            modelPath=args.model_path,
            numClasses=2,
            confidence=args.confidence,
        )
        
        if args.batch:
            # Batch processing of directory
            if not os.path.isdir(args.image_path):
                raise ValueError("For batch processing, image_path must be a directory")
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(args.image_path, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(args.image_path, f"*{ext.upper()}")))
            
            print(f"Found {len(image_files)} images to process")
            
            for i, image_file in enumerate(image_files, 1):
                print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_file)}")
                try:
                    mask, coloredMask = segformer.predict(
                        imagePath=image_file,
                        outputDir=args.output_dir,
                        visualize=not args.no_visualize
                    )
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    continue
        else:
            # Single image inference
            mask, coloredMask = segformer.predict(
                imagePath=args.image_path,
                outputDir=args.output_dir,
                visualize=not args.no_visualize
            )
            print("Inference completed!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure your model and image paths are correct!")