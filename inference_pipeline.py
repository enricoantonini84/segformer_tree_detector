# inference pipeline is directly derived from my other repo https://github.com/enricoantonini84/satellite_tree_detector
# this one is only compatbile with SegFormer, and it's not available in the same repo of the others due
# to licencing issues
# this script predicts tree presence with the help of ingerence.py
# see readme! (is AI generated, trust me, Anthropic english is better than mine)

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import glob
import cv2
import numpy as np
import rasterio

# import existing inference modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'sf'))

from sf.inference import SegFormerInference

# import GeoJSON conversion functions
sys.path.append(os.path.dirname(__file__))
from json_to_geojson import findJsonFiles, mergeJsonToGeoJson

# logging, maybe useless? should be removed to make the script smaller?
def segformerDetector(imagePath: str,
                     modelPath: str,
                     conf: float = 0.5,
                     fromGeoTIFF: bool = True,
                     outputFolder: Optional[str] = None,
                     saveAnnotated: bool = True) -> Dict[str, Any]:
    
    # initialize Segformer inference
    segformer = SegFormerInference(
        modelPath=modelPath,
        numClasses=2,
        confidence=conf
    )
    
    # read image and get geospatial info if it's a GeoTIFF
    geospatialInfo = None
    if fromGeoTIFF and imagePath.lower().endswith(('.tif', '.tiff')):
        try:
            with rasterio.open(imagePath) as src:
                geospatialInfo = {
                    'transform': src.transform,
                    'crs': src.crs.to_string() if src.crs else None,
                    'bounds': src.bounds,
                    'shape': src.shape
                }
        except Exception as e:
            print(f"Warning: Could not read GeoTIFF metadata: {e}")
            fromGeoTIFF = False
    
    try:
        mask, coloredMask = segformer.predict(
            imagePath=imagePath,
            outputDir=outputFolder if saveAnnotated else None,
            visualize=False  # We'll handle visualization separately
        )
        
        # segmentation mask to polygons conversion
        polygons, geoPolygons = maskToPolygons(mask, geospatialInfo)
        
        # stats calculation
        treePixels = np.sum(mask == 1)
        totalPixels = mask.size
        treeCoveragePercent = (treePixels / totalPixels) * 100
        
        # result dictionary in pipeline format
        result = {
            'image_path': imagePath,
            'model_type': 'segformer',
            'model_path': modelPath,
            'confidence_threshold': conf,
            'trees_detected': len(polygons),
            'tree_coverage_pixels': int(treePixels),
            'total_pixels': int(totalPixels),
            'tree_coverage_percent': float(treeCoveragePercent),
            'polygons': polygons,
            'num_detections': len(polygons)
        }
        
        # geospatial information if available
        if geospatialInfo:
            result.update({
                'transform': list(geospatialInfo['transform']),
                'crs': geospatialInfo['crs'],
                'bounds': list(geospatialInfo['bounds']),
                'geo_polygons': geoPolygons
            })
        
        # Save annotated image if requested
        if saveAnnotated and outputFolder:
            saveSegformerAnnotation(imagePath, coloredMask, outputFolder)
        
        print(f"Segformer detection completed: {len(polygons)} tree regions found")
        print(f"Tree coverage: {treeCoveragePercent:.2f}% ({treePixels:,}/{totalPixels:,} pixels)")
        
        return result
        
    except Exception as e:
        print(f"Error during Segformer inference: {e}")
        return {
            'error': str(e),
            'image_path': imagePath,
            'model_type': 'segformer',
            'trees_detected': 0,
            'num_detections': 0
        }
# convert segmentation mask to polygon contours.
def maskToPolygons(mask: np.ndarray,
                   geospatialInfo: Optional[Dict] = None,
                   minArea: int = 50) -> Tuple[List[List], List[Dict]]:
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    geoPolygons = []
    
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < minArea:
            continue
            
        # Simplify contour to reduce points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplifiedContour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to polygon coordinates
        if len(simplifiedContour) >= 3:  # Need at least 3 points for polygon
            pixelCoords = [(int(point[0][0]), int(point[0][1])) for point in simplifiedContour]
            polygons.append(pixelCoords)
            
            # Transform to geographic coordinates if geospatial info available
            if geospatialInfo and geospatialInfo.get('transform'):
                geoCoords = []
                transform = geospatialInfo['transform']
                
                for x, y in pixelCoords:
                    lon, lat = rasterio.transform.xy(transform, y, x)
                    geoCoords.append([lon, lat])
                
                # Calculate geographic center
                if geoCoords:
                    centerLon = sum(coord[0] for coord in geoCoords) / len(geoCoords)
                    centerLat = sum(coord[1] for coord in geoCoords) / len(geoCoords)
                    
                    geoPolygons.append({
                        'polygon': geoCoords,
                        'geo_center': {'lon': centerLon, 'lat': centerLat}
                    })
                else:
                    geoPolygons.append({'polygon': [], 'geo_center': {'lon': None, 'lat': None}})
            else:
                geoPolygons.append({'polygon': [], 'geo_center': {'lon': None, 'lat': None}})
    
    return polygons, geoPolygons

# save annotatated result after sefgformer prediction
def saveSegformerAnnotation(imagePath: str,
                            coloredMask: np.ndarray,
                            outputFolder: str):
    """Save annotated Segformer results."""
    
    os.makedirs(outputFolder, exist_ok=True)
    
    # Read original image
    originalImage = cv2.imread(imagePath)
    if originalImage is None:
        print(f"Warning: Could not read original image for annotation: {imagePath}")
        return
    
    # Resize colored mask to match original image size
    if coloredMask.shape[:2] != originalImage.shape[:2]:
        coloredMask = cv2.resize(
            coloredMask,
            (originalImage.shape[1], originalImage.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create overlay
    alpha = 0.6
    # Convert coloredMask from RGB to BGR to match OpenCV's BGR format
    coloredMaskBgr = cv2.cvtColor(coloredMask, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(originalImage, 1-alpha, coloredMaskBgr, alpha, 0)
    
    # Save annotated image
    baseName = os.path.splitext(os.path.basename(imagePath))[0]
    
    # Save with GeoTIFF format if input was GeoTIFF
    if imagePath.lower().endswith(('.tif', '.tiff')):
        outputPath = os.path.join(outputFolder, f"{baseName}_segformer_annotated.tif")
        
        # Try to preserve geospatial info
        try:
            with rasterio.open(imagePath) as src:
                profile = src.profile.copy()
                profile.update({
                    'dtype': rasterio.uint8,
                    'count': 3
                })
                
                with rasterio.open(outputPath, 'w', **profile) as dst:
                    # Convert BGR to RGB and write
                    overlayRgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    for i in range(3):
                        dst.write(overlayRgb[:, :, i], i + 1)
                        
        except Exception as e:
            # Fallback to regular image save
            print(f"Warning: Could not save as GeoTIFF, saving as PNG: {e}")
            outputPath = os.path.join(outputFolder, f"{baseName}_segformer_annotated.png")
            # Convert BGR to RGB before saving as PNG
            overlayRgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.imwrite(outputPath, overlayRgb)
    else:
        # Save as PNG for regular images
        outputPath = os.path.join(outputFolder, f"{baseName}_segformer_annotated.png")
        # Convert BGR to RGB before saving as PNG
        overlayRgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        cv2.imwrite(outputPath, overlayRgb)
    
    print(f"Segformer annotated image saved: {outputPath}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TreeDetectionPipeline:
    
    # init modeltype and other parameters for segformer inference
    def __init__(self, modelType: str, modelPath: str, confidence: float = 0.5):
        self.modelType = modelType.lower()
        self.modelPath = modelPath
        self.confidence = confidence
        
        if self.modelType != 'segformer':
            raise ValueError("modelType must be 'segformer'")
        
        if not os.path.exists(self.modelPath):
            raise FileNotFoundError(f"Model file not found: {self.modelPath}")
            
        logger.info(f"Initialized {self.modelType.upper()} pipeline with confidence {confidence}")
    
    # called to predict trees presence on a single file.
    # this calls the yolo detector or the detectree2 detector declared in other folders
    def detectSingleImage(self, imagePath: str, outputFolder: str,
                          saveImages: bool = True,
                          saveJson: bool = True) -> Dict[str, Any]:
        logger.info(f"Processing single image: {imagePath}")
        
        os.makedirs(outputFolder, exist_ok=True)
        
        # geotiff is true by default, other formats are not supported in this pipeline
        try:
            # segformer inference
            results = segformerDetector(
                imagePath=imagePath,
                modelPath=self.modelPath,
                conf=self.confidence,
                fromGeoTIFF=True,
                outputFolder=os.path.join(outputFolder, "annotated") if saveImages else None,
                saveAnnotated=saveImages
            )
            
            # user can chooose if he want to save json or not, and also if he want to save annotated
            # images. if he choose not to save json, it would be impossibile to generate GeoJson at the end
            if saveJson:
                jsonPath = Path(outputFolder) / "json" / f"{Path(imagePath).stem}_results.json"
                with open(jsonPath, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"JSON results saved to: {jsonPath}")
            
            if saveImages:
                logger.info(f"Annotated images should be saved in: {os.path.join(outputFolder, 'annotated')}")
            else:
                logger.info("Saving annotated images was disabled")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {imagePath}: {str(e)}")
            return {"error": str(e), "image_path": imagePath}
    
    # process an entire folter of tiles
    def processTilesFolder(self,
                           inputFolder: str,
                           outputFolder: str,
                           saveImages: bool = True,
                           saveJson: bool = True) -> List[Dict[str, Any]]:
        logger.info(f"Processing tiles folder: {inputFolder}")
        
        #we don't want any non geotiff file
        geotiffExtensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        
        tiles = []
        for extension in geotiffExtensions:
            patternPath = os.path.join(inputFolder, extension)
            foundFiles = glob.glob(patternPath)
            tiles.extend(foundFiles)
        
        results = []
        for i, tilePath in enumerate(tiles, 1):
            logger.info(f"Processing tile {i}/{len(tiles)}: {os.path.basename(tilePath)}")
            
            result = self.detectSingleImage(
                imagePath=tilePath,
                outputFolder=outputFolder,
                saveImages=saveImages,
                saveJson=saveJson
            )
            
            results.append(result)
            
            if i % 10 == 0 or i == len(tiles):
                logger.info(f"Completed {i}/{len(tiles)} tiles")
        
        return results
    
    # this can convert the json woth detected trees into a geojson
    def convertToGeojson(self, pipelineSummary: Dict[str, Any], outputFolder: str) -> bool:
        logger.info("Starting JSON to GeoJSON conversion...")
        
        try:
            jsonFiles = findJsonFiles(os.path.join(outputFolder, "json"))
            
            if not jsonFiles:
                logger.warning(f"No JSON files found in {outputFolder}")
                return False
            
            logger.info(f"Found {len(jsonFiles)} JSON files to convert")
            
            inputBasename = Path(pipelineSummary.get("input_path", "detections")).stem
            geojsonFilename = f"{inputBasename}_detections.geojson"
            geojsonPath = os.path.join(outputFolder, "json", geojsonFilename)
            
            geoJson = mergeJsonToGeoJson(jsonFiles, geojsonPath)
            
            # geojson is composed by a number of features, that are polygons detected by
            # a ML algorythm
            if geoJson and geoJson.get('features'):
                logger.info(f"Successfully created GeoJSON with {len(geoJson['features'])} features")
                pipelineSummary["geojson_output"] = geojsonPath
                pipelineSummary["total_geojson_features"] = len(geoJson['features'])
                return True
            else:
                logger.error("Failed to create GeoJSON or no features found")
                return False
                
        except Exception as e:
            logger.error(f"Error during GeoJSON conversion: {str(e)}")
            return False
    
    # that's the real deal
    def runPipeline(self,
                    inputPath: str,
                    outputFolder: str = "output",
                    saveImages: bool = True,
                    saveJson: bool = True,
                    createGeojson: bool = True) -> Dict[str, Any]:
        logger.info("-"*50)
        logger.info("STARTING TREE DETECTION PIPELINE")
        logger.info("-"*50)
        
        # Create output folder and subfolders
        os.makedirs(outputFolder, exist_ok=True)
        os.makedirs(os.path.join(outputFolder, "json"), exist_ok=True)
        os.makedirs(os.path.join(outputFolder, "annotated"), exist_ok=True)
        
        pipelineSummary = {
            "input_path": inputPath,
            "model_type": self.modelType,
            "model_path": self.modelPath,
            "confidence": self.confidence,
            "output_folder": outputFolder,
            "results": []
        }
        
        try:
            if os.path.isfile(inputPath):
                # process single GeoTIFF file
                result = self.detectSingleImage(
                    imagePath=inputPath,
                    outputFolder=outputFolder,
                    saveImages=saveImages,
                    saveJson=saveJson
                )
                results = [result]
                pipelineSummary["processing_mode"] = "single_geotiff"
                    
            elif os.path.isdir(inputPath):
                # process folder of GeoTIFF tiles
                results = self.processTilesFolder(
                    inputFolder=inputPath,
                    outputFolder=outputFolder,
                    saveImages=saveImages,
                    saveJson=saveJson
                )
                pipelineSummary["processing_mode"] = "geotiff_tiles_folder"
                
            else:
                raise FileNotFoundError(f"Input path not found: {inputPath}")
            
            pipelineSummary["results"] = results
            
            # Calculate summary statistics
            successfulResults = [r for r in results if "error" not in r]
            pipelineSummary["total_processed"] = len(results)
            pipelineSummary["successful_processed"] = len(successfulResults)
            pipelineSummary["failed_processed"] = len(results) - len(successfulResults)
            
            if successfulResults:
                totalDetections = sum(
                    r.get("trees_detected", r.get("num_detections", 0))
                    for r in successfulResults
                )
                pipelineSummary["total_detections"] = totalDetections
            
            # convert JSON results to GeoJSON if requested and JSON files were saved
            if createGeojson and saveJson and successfulResults:
                geojsonSuccess = self.convertToGeojson(pipelineSummary, outputFolder)
                pipelineSummary["geojson_conversion_success"] = geojsonSuccess
            elif createGeojson and not saveJson:
                logger.info("GeoJSON conversion skipped because JSON saving is disabled")
            elif createGeojson and not successfulResults:
                logger.info("GeoJSON conversion skipped because no successful detections were made")
            
            # save pipeline summary
            summaryPath = os.path.join(outputFolder, "json", "pipeline_summary.json")
            with open(summaryPath, 'w') as f:
                json.dump(pipelineSummary, f, indent=2)
            
            logger.info("="*50)
            logger.info("PIPELINE EXECUTION COMPLETED")
            logger.info(f"Total processed: {pipelineSummary['total_processed']}")
            logger.info(f"Successful: {pipelineSummary['successful_processed']}")
            logger.info(f"Failed: {pipelineSummary['failed_processed']}")
            if "total_detections" in pipelineSummary:
                logger.info(f"Total detections: {pipelineSummary['total_detections']}")
            if "total_geojson_features" in pipelineSummary:
                logger.info(f"GeoJSON features: {pipelineSummary['total_geojson_features']}")
            if "geojson_output" in pipelineSummary:
                logger.info(f"GeoJSON saved to: {pipelineSummary['geojson_output']}")
            logger.info(f"Annotated tiles saved to: {os.path.join(outputFolder, 'annotated')}")
            logger.info(f"Results saved to: {outputFolder}")
            logger.info("="*50)
            
            return pipelineSummary
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            pipelineSummary["error"] = str(e)
            return pipelineSummary

# if you pass this script in a prompt it will say that this is ai generated because
# all of those help messages written seriously good... :)
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input path (single GeoTIFF file or folder of tiles)')
    parser.add_argument('--model', '-m', default='segformer', choices=['segformer'],
                       help='Model type to use for detection (default: segformer)')
    parser.add_argument('--model-path', '-p', required=True,
                       help='Path to model file')
    parser.add_argument('--output', '-o', default='output',
                       help='Output folder (default: output)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip saving annotated images')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip saving JSON detection data')
    parser.add_argument('--no-geojson', action='store_true',
                       help='Skip converting JSON results to GeoJSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        pipeline = TreeDetectionPipeline(
            modelType=args.model,
            modelPath=args.model_path,
            confidence=args.confidence
        )
        
        summary = pipeline.runPipeline(
            inputPath=args.input,
            outputFolder=args.output,
            saveImages=not args.no_images,
            saveJson=not args.no_json,
            createGeojson=not args.no_geojson
        )
        
        if "error" in summary:
            sys.exit(1)
        else:
            sys.exit(0)  # success
            
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()