# this is a segformer training script, using huggingface libraries
# accepts in input a folder containing the dataset of masks and images
# because of the hf library helper processor, theres no need to uniform
# image size
#
# check main for the arguments

import os
import cv2
import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
    DefaultDataCollator
)
# processor init
imageProcessor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")

# data loade function
def loadDataSimple(dataDir):
    imagesDir = os.path.join(dataDir, 'images')
    masksDir = os.path.join(dataDir, 'masks')
    
    imagePaths = [os.path.join(imagesDir, f) for f in sorted(os.listdir(imagesDir))]
    
    # mask matching
    maskPaths = []
    for imgPath in imagePaths:
        baseName = os.path.splitext(os.path.basename(imgPath))[0]
        # Try common mask naming patterns
        for ext in ['.png', '.jpg']:
            maskPath = os.path.join(masksDir, f"{baseName}_mask{ext}")
            if os.path.exists(maskPath):
                maskPaths.append(maskPath)
                break
            maskPath = os.path.join(masksDir, f"{baseName}{ext}")
            if os.path.exists(maskPath):
                maskPaths.append(maskPath)
                break
        else:
            # dummy mask if none found
            maskPaths.append(None)
    
    return imagePaths, maskPaths

# this function use the Hugging face preprocessor to normalize the images
def preprocessData(imagePaths, maskPaths):
    
    def processExample(imagePath, maskPath):
        # load image
        image = cv2.imread(imagePath, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # load mask
        if maskPath and os.path.exists(maskPath):
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # Binary mask
        else:
            # create dummy mask with same dimensions as image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # HF processor for both image AND segmentation map - handles resizing automatically!
        processed = imageProcessor(
            images=image,
            segmentation_maps=mask,
            return_tensors="np"
        )
        
        return {
            "pixel_values": processed["pixel_values"][0],  # Remove batch dim
            "labels": processed["labels"][0].astype(np.int64)  # Remove batch dim
        }
    
    # process all data
    data = [processExample(img, mask) for img, mask in zip(imagePaths, maskPaths)]
    
    # convert to HuggingFace Dataset
    return HFDataset.from_list(data)

# Ultra-simple training function - everything in one place!
def trainSegformer(
    dataDir,
    outputDir="./results",
    epochs=40,
    learningRate=0.00006,
    batchSize=2
):
    
    print("Starting SegFormer training...")
    
    # load data and start preprocessing
    imagePaths, maskPaths = loadDataSimple(dataDir)
    dataset = preprocessData(imagePaths, maskPaths)
    
    # split data with huggingface utils
    splits = dataset.train_test_split(test_size=0.3, seed=42)
    valTest = splits['test'].train_test_split(test_size=0.33, seed=42)
    
    trainDataset = splits['train']
    valDataset = valTest['train']
    testDataset = valTest['test']
    
    # check if GPU is available (you should have it, otherwise it will take YEARS!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3",
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # training arguments - optimized for maximum GPU performance!
    trainingArgs = TrainingArguments(
        output_dir=outputDir,
        learning_rate=learningRate,
        per_device_train_batch_size=batchSize,
        per_device_eval_batch_size=batchSize,
        gradient_accumulation_steps=4,  # Effective batch size = batchSize * 4
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=[],  # No logging
        use_cpu=not torch.cuda.is_available(),  # Explicitly use GPU if available
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory for GPU
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,  # Parallel data loading
        fp16=torch.cuda.is_available(),  # Mixed precision for speed
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # Faster optimizer
        warmup_ratio=0.1,  # Learning rate warmup
        weight_decay=0.01,  # Better regularization
        logging_steps=50,  # Less frequent logging
    )
    
    trainer = Trainer(
        model=model,
        args=trainingArgs,
        train_dataset=trainDataset,
        eval_dataset=valDataset,
        data_collator=DefaultDataCollator(),  # Built-in collator!
    )
    
    print("Training started...")
    trainer.train()
    
    print("Final evaluation...")
    results = trainer.evaluate(testDataset)
    print(f"Test Results: {results}")
    
    trainer.save_model()
    print(f"Model saved to {outputDir}")
    
    return trainer, results

# MAIN
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SegFormer Training Script")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing images and masks folders')
    parser.add_argument('--output_dir', type=str, default='./segformer_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00006, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    
    args = parser.parse_args()
    
    print("Starting ultra-simplified SegFormer training!")
    
    # this is it - one function call does everything!
    trainer, results = trainSegformer(
        dataDir=args.data_dir,
        outputDir=args.output_dir,
        epochs=args.epochs,
        learningRate=args.learning_rate,
        batchSize=args.batch_size
    )
    
    print("Training completed successfully!")
    print(f"Final metrics: {results}")