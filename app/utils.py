import json
import os
from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import shape, Point, LineString
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class GeoJSONHelper:
    def __init__(self):
        self.barangays = []
        self.waterways = []
        self._load_geojson_data()
    
    def get_property_value(self, properties: dict, possible_keys: list, default=None):
        """Get property value trying multiple possible keys"""
        for key in possible_keys:
            if key in properties and properties[key] is not None and str(properties[key]).strip():
                return properties[key]
        return default
    
    def _load_geojson_data(self):
        """Load GeoJSON data for barangays and waterways"""
        try:
            # Load barangay polygons
            barangay_path = os.path.join(os.path.dirname(__file__), "..", "data", "dasmabarangays.geojson")
            
            if not os.path.exists(barangay_path):
                logger.warning(f"Barangay GeoJSON file not found at {barangay_path}")
                self._create_dummy_data()
                return
            
            with open(barangay_path, 'r', encoding='utf-8') as f:
                barangay_geojson = json.load(f)
            
            logger.info(f"Loading {len(barangay_geojson.get('features', []))} barangay features")
            
            for i, feature in enumerate(barangay_geojson["features"]):
                try:
                    polygon = shape(feature["geometry"])
                    properties = feature.get("properties", {})
                    
                    # Try multiple possible name fields
                    name = self.get_property_value(
                        properties, 
                        ["NAME", "name", "barangay_name", "BARANGAY", "Barangay", "BRGY_NAME"],
                        None  # Don't set default here
                    )
                    
                    # Skip if no valid name found
                    if not name or str(name).strip() == "" or str(name).strip().lower() in ["unknown", "null", "none"]:
                        logger.warning(f"Skipping barangay feature {i} with invalid name: {name}")
                        continue
                    
                    # Ensure barangay_id exists
                    barangay_id = self.get_property_value(
                        properties,
                        ["barangay_id", "id", "OBJECTID", "FID", "ID"],
                        i + 1
                    )
                    
                    # Add barangay_id to properties if not present
                    if "barangay_id" not in properties:
                        properties["barangay_id"] = barangay_id
                    
                    self.barangays.append({
                        "name": str(name).strip(),
                        "polygon": polygon,
                        "properties": properties
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing barangay feature {i}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(self.barangays)} barangays")
            
            # Load waterways
            waterway_path = os.path.join(os.path.dirname(__file__), "..", "data", "waterway.geojson")
            
            if os.path.exists(waterway_path):
                with open(waterway_path, 'r', encoding='utf-8') as f:
                    waterway_geojson = json.load(f)
                
                for feature in waterway_geojson["features"]:
                    try:
                        geometry = shape(feature["geometry"])
                        self.waterways.append({
                            "geometry": geometry,
                            "properties": feature.get("properties", {})
                        })
                    except Exception as e:
                        logger.error(f"Error processing waterway feature: {e}")
                        continue
                
                logger.info(f"Successfully loaded {len(self.waterways)} waterways")
            else:
                logger.warning(f"Waterway GeoJSON file not found at {waterway_path}")
                self._create_dummy_waterways()
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid GeoJSON format: {e}")
            self._create_dummy_data()
        except FileNotFoundError as e:
            logger.error(f"GeoJSON file not found: {e}")
            self._create_dummy_data()
        except Exception as e:
            logger.error(f"Error loading GeoJSON data: {e}")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy GeoJSON data for testing"""
        logger.info("Creating dummy barangay data for testing")
        
        # Create multiple dummy barangays for better testing
        from shapely.geometry import Polygon
        
        dummy_barangays_data = [
            {
                "name": "San Simon",
                "coords": [(120.930, 14.325), (120.940, 14.325), (120.940, 14.335), (120.930, 14.335), (120.930, 14.325)],
                "id": 1
            },
            {
                "name": "Burol Main", 
                "coords": [(120.940, 14.325), (120.950, 14.325), (120.950, 14.335), (120.940, 14.335), (120.940, 14.325)],
                "id": 2
            },
            {
                "name": "Zone IV",
                "coords": [(120.930, 14.335), (120.940, 14.335), (120.940, 14.345), (120.930, 14.345), (120.930, 14.335)],
                "id": 3
            }
        ]
        
        self.barangays = []
        for brgy_data in dummy_barangays_data:
            polygon = Polygon(brgy_data["coords"])
            self.barangays.append({
                "name": brgy_data["name"],
                "polygon": polygon,
                "properties": {
                    "NAME": brgy_data["name"],
                    "barangay_id": brgy_data["id"],
                    "OBJECTID": brgy_data["id"]
                }
            })
        
        self._create_dummy_waterways()
    
    def _create_dummy_waterways(self):
        """Create dummy waterway data"""
        # Dummy waterway
        from shapely.geometry import LineString
        dummy_waterway = LineString([
            (120.935, 14.330),
            (120.937, 14.332),
            (120.939, 14.334)
        ])
        
        self.waterways = [{
            "geometry": dummy_waterway,
            "properties": {"name": "Dummy Creek"}
        }]
    
    def is_inside_dasma(self, lat: float, lng: float) -> Tuple[bool, Optional[str]]:
        """
        Check if coordinates are inside Dasmariñas
        Args:
            lat: Latitude
            lng: Longitude
        Returns:
            Tuple of (is_inside, barangay_name)
        """
        try:
            point = Point(lng, lat)  # Shapely expects (x=lng, y=lat)
            
            for barangay in self.barangays:
                try:
                    if barangay["polygon"].contains(point):
                        return True, barangay["name"]
                except Exception as e:
                    logger.warning(f"Error checking point in polygon for {barangay.get('name', 'Unknown')}: {e}")
                    continue
            
            return False, None
        except Exception as e:
            logger.error(f"Error in is_inside_dasma: {e}")
            return False, None
    
    def get_barangay_name(self, lat: float, lng: float) -> Optional[str]:
        """Get barangay name for given coordinates"""
        try:
            _, barangay_name = self.is_inside_dasma(lat, lng)
            if barangay_name and isinstance(barangay_name, str):
                return barangay_name
            return None
        except Exception as e:
            logger.error(f"Error getting barangay name: {e}")
            return None
    
    def get_nearest_waterway_distance(self, lat: float, lng: float) -> float:
        """
        Calculate distance to nearest waterway
        Args:
            lat: Latitude
            lng: Longitude
        Returns:
            Distance in meters
        """
        try:
            point = Point(lng, lat)
            min_distance = float('inf')
            
            for waterway in self.waterways:
                try:
                    geometry = waterway["geometry"]
                    distance = geometry.distance(point)
                    
                    # Convert to meters (approximate)
                    distance_meters = self._degrees_to_meters(distance)
                    min_distance = min(min_distance, distance_meters)
                except Exception as e:
                    logger.warning(f"Error calculating distance to waterway: {e}")
                    continue
            
            return round(min_distance, 2) if min_distance != float('inf') else 1000.0
        except Exception as e:
            logger.error(f"Error in get_nearest_waterway_distance: {e}")
            return 1000.0
    
    def _degrees_to_meters(self, degrees: float) -> float:
        """Convert degrees to meters (approximate)"""
        # 1 degree ≈ 111,320 meters at the equator
        return degrees * 111320
    
    def get_all_barangays(self) -> List[dict]:
        """Get list of all barangays with their GeoJSON data"""
        result = []
        for barangay in self.barangays:
            try:
                # Check if it's a Polygon (has exterior) or other geometry
                if hasattr(barangay["polygon"], 'exterior'):
                    # It's a Polygon
                    coords = list(barangay["polygon"].exterior.coords)
                elif hasattr(barangay["polygon"], 'coords'):
                    # It's a LineString or other geometry with coords
                    coords = list(barangay["polygon"].coords)
                else:
                    logger.warning(f"Unknown geometry type for barangay {barangay['name']}")
                    continue
                
                result.append({
                    "name": barangay["name"],
                    "properties": barangay["properties"],
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                })
            except Exception as e:
                logger.error(f"Error processing barangay {barangay.get('name', 'Unknown')}: {e}")
                continue
        return result
    
    def get_barangay_centroid(self, barangay_geojson: dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the centroid coordinates of a barangay from GeoJSON
        Args:
            barangay_geojson: GeoJSON feature of the barangay
        Returns:
            Tuple of (lat, lng) or (None, None) if not found
        """
        try:
            if 'geometry' in barangay_geojson and 'coordinates' in barangay_geojson['geometry']:
                # Get the first ring of the polygon
                coords = barangay_geojson['geometry']['coordinates'][0]
                
                # Calculate centroid
                lng_sum = sum(coord[0] for coord in coords if len(coord) >= 2)
                lat_sum = sum(coord[1] for coord in coords if len(coord) >= 2)
                count = len([coord for coord in coords if len(coord) >= 2])
                
                if count > 0:
                    centroid_lng = lng_sum / count
                    centroid_lat = lat_sum / count
                    return centroid_lat, centroid_lng
        except Exception as e:
            logger.error(f"Error calculating centroid: {e}")
        
        return None, None
    
    def get_barangay_id_from_coordinates(self, lat: float, lng: float) -> Optional[int]:
        """
        Get barangay_id from coordinates using GeoJSON data
        Args:
            lat: Latitude
            lng: Longitude
        Returns:
            Barangay ID or None if not found
        """
        try:
            point = Point(lng, lat)  # Shapely expects (x=lng, y=lat)
            
            for i, barangay in enumerate(self.barangays):
                try:
                    if barangay["polygon"].contains(point):
                        # Try to get barangay_id from properties
                        properties = barangay.get("properties", {})
                        barangay_id = self.get_property_value(
                            properties,
                            ["barangay_id", "id", "OBJECTID", "FID"],
                            i + 1  # fallback to index + 1
                        )
                        
                        # Convert to int if it's a string
                        if isinstance(barangay_id, str):
                            try:
                                return int(barangay_id)
                            except ValueError:
                                logger.warning(f"Invalid barangay_id format: {barangay_id}")
                                return i + 1
                        
                        return int(barangay_id) if barangay_id is not None else i + 1
                        
                except Exception as e:
                    logger.warning(f"Error checking point for barangay {barangay.get('name', 'Unknown')}: {e}")
                    continue
            
            return None
        except Exception as e:
            logger.error(f"Error in get_barangay_id_from_coordinates: {e}")
            return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    Returns:
        Distance in meters
    """
    try:
        R = 6371000  # Earth's radius in meters
        
        # Convert to radians
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    except Exception as e:
        logger.error(f"Error calculating haversine distance: {e}")
        return 0.0

# Global instance
geojson_helper = GeoJSONHelper()

# Convenience functions for backward compatibility
def is_inside_dasma(lat: float, lng: float) -> Tuple[bool, Optional[str]]:
    """Check if coordinates are inside Dasmariñas"""
    return geojson_helper.is_inside_dasma(lat, lng)

def get_barangay_name(lat: float, lng: float) -> Optional[str]:
    """Get barangay name for given coordinates"""
    return geojson_helper.get_barangay_name(lat, lng)

def get_nearest_waterway_distance(lat: float, lng: float) -> float:
    """Calculate distance to nearest waterway"""
    return geojson_helper.get_nearest_waterway_distance(lat, lng)

def get_barangay_id_from_coordinates(lat: float, lng: float) -> Optional[int]:
    """Get barangay_id from coordinates"""
    return geojson_helper.get_barangay_id_from_coordinates(lat, lng)

def get_barangay_centroid(barangay_geojson: dict) -> Tuple[Optional[float], Optional[float]]:
    """Get barangay centroid coordinates"""
    return geojson_helper.get_barangay_centroid(barangay_geojson);