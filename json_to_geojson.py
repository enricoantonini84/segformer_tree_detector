import os
import json
import argparse
import glob


#convert segformer detection to geojson feature
def segformerToGeoJsonFeature(detection, detectionIdx, imageInfo=None):
    """Convert Segformer polygon detection to GeoJSON feature"""
    # Use geoPolygons if available, otherwise pixel polygons
    if imageInfo and imageInfo.get('geo_polygons') and detectionIdx < len(imageInfo['geo_polygons']):
        geoPolygon = imageInfo['geo_polygons'][detectionIdx]
        coords = geoPolygon.get('polygon', [])
        centerLon = geoPolygon.get('geo_center', {}).get('lon')
        centerLat = geoPolygon.get('geo_center', {}).get('lat')
        
        # Coordinates are already in geographic format
        if coords:
            geoCoords = coords
        else:
            geoCoords = []
    else:
        # Fallback to pixel polygons with transformation
        coords = detection
        if coords and imageInfo and imageInfo.get('transform'):
            transform = imageInfo['transform']
            geoCoords = []
            for x, y in coords:
                lon = transform[0] + x * transform[1] + y * transform[2]
                lat = transform[3] + x * transform[4] + y * transform[5]
                geoCoords.append([lon, lat])
        else:
            geoCoords = coords if coords else []
        centerLon = None
        centerLat = None
    
    # Ensure polygon is closed
    if geoCoords and len(geoCoords) > 0 and geoCoords[0] != geoCoords[-1]:
        geoCoords.append(geoCoords[0])
    
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [geoCoords] if geoCoords else []
        },
        "properties": {
            "id": detectionIdx + 1,
            "detection_type": "segformer_polygon",
            "center_lon": centerLon,
            "center_lat": centerLat
        }
    }
    
    if imageInfo:
        feature['properties']['source_image'] = imageInfo.get('image_path')
        feature['properties']['crs'] = imageInfo.get('crs')
        if imageInfo.get('tree_coverage_percent'):
            feature['properties']['tree_coverage_percent'] = imageInfo['tree_coverage_percent']
    
    return feature

#parse json detection file and convert to geojson features
def parseJsonFile(jsonPath):
    try:
        with open(jsonPath, 'r') as f:
            data = json.load(f)
        
        features = []
        
        # Process Segformer polygon data
        if 'polygons' in data:
            polygons = data['polygons']
            for i, polygon in enumerate(polygons):
                if polygon:  # skip empty polygons
                    feature = segformerToGeoJsonFeature(polygon, i, data)
                    features.append(feature)
        
        print(f"Processed {jsonPath}: {len(features)} detections")
        return features
        
    except Exception as e:
        print(f"Error processing {jsonPath}: {str(e)}")
        return []

#find all json files in a folder
def findJsonFiles(folderPath):
    jsonPatterns = ['*_results.json', '*.json']
    jsonFiles = []
    
    for pattern in jsonPatterns:
        jsonFiles.extend(glob.glob(os.path.join(folderPath, pattern)))
    
    return sorted(list(set(jsonFiles)))

#merge multiple json files into single geojson
def mergeJsonToGeoJson(jsonFiles, outputPath):
    allFeatures = []
    
    for jsonFile in jsonFiles:
        features = parseJsonFile(jsonFile)
        allFeatures.extend(features)
    
    #create geojson structure with proper CRS specification
    geoJson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        },
        "features": allFeatures,
        "properties": {
            "total_detections": len(allFeatures),
            "source_files": len(jsonFiles),
            "generated_by": "satellite_tree_cover_detection_pipeline"
        }
    }
    
    with open(outputPath, 'w') as f:
        json.dump(geoJson, f, indent=2)
    
    print(f"Created GeoJSON with {len(allFeatures)} features from {len(jsonFiles)} files")
    print(f"Output saved to: {outputPath}")
    
    return geoJson

#process folder and convert all json to geojson
def processFolder(inputFolder, outputPath):
    if not os.path.exists(inputFolder):
        print(f"Error: Input folder not found: {inputFolder}")
        return False

    jsonFiles = findJsonFiles(inputFolder)
    
    if not jsonFiles:
        print(f"No JSON files found in {inputFolder}")
        return False
    
    print(f"Found {len(jsonFiles)} JSON files to process")

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    geoJson = mergeJsonToGeoJson(jsonFiles, outputPath)
    
    return len(geoJson['features']) > 0

def main():
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', help='Input folder containing JSON files')
    group.add_argument('--files', nargs='+', help='List of specific JSON files to merge')
    
    parser.add_argument('--output', '-o', required=True, help='Output GeoJSON file path')
    
    args = parser.parse_args()
    
    try:
        if args.input:
            success = processFolder(args.input, args.output)
        else:
            success = len(mergeJsonToGeoJson(args.files, args.output)['features']) > 0
            
        if success:
            print("JSON to GeoJSON conversion completed successfully")
        else:
            print("Failed to convert JSON files")
            
    except Exception as e:
        print(f"Conversion failed: {str(e)}")

if __name__ == "__main__":
    main()