from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from datetime import datetime
import logging
import random
import math

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="Dengue Risk Prediction API",
    description="Potential breeding sites for Dasmariñas",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Imports we actually need
from app.utils import (
    is_inside_dasma, 
    get_nearest_waterway_distance, 
)
from app.supabase_client import supabase_client
from app.weather_client import weather_client


@app.get("/")
async def root() -> Dict[str, Any]:
        return {
        "status": "healthy",
        "message": "Dengue Prediction API is running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/predict/potential-breeding-sites")
async def predict_potential_breeding_sites(
    radius_meters: float = 150,
    max_points_per_seed: int = 8,
    min_probability: float = 0.6,
    days_cases: int = 30,
    days_sites: int = 30,
    predicted_days: int = 14,
) -> List[Dict[str, Any]]:
    """
    Predict potential future breeding sites across Dasmariñas using:
    - recent confirmed dengue_cases (seed weight = 1.0)
    - recent breeding_sites (seed weight = 0.6)
    - weather (rainfall, humidity, temperature near-optimal band)
    - proximity to waterways (closer => higher)

    Returns flat list of { longitude, latitude, risk_level, probability, predicted_days }.
    Default predicted_days = 14 (2 weeks forecast period).
    """
    try:
        # 1) Collect seeds across the city (no barangay dependence)
        cases_rows = await supabase_client.fetch_recent_dengue_cases(days_cases)
        sites_rows = await supabase_client.fetch_recent_breeding_sites(days_sites)

        recent_cases = []  # (lat, lng)
        for r in cases_rows:
            lat = (r.get("resident") or {}).get("latitude")
            lng = (r.get("resident") or {}).get("longitude")
            if lat and lng:
                if is_inside_dasma(float(lat), float(lng))[0]:
                    recent_cases.append((float(lat), float(lng)))

        recent_sites = []  # (lat, lng)
        for s in sites_rows:
            lat = s.get("latitude")
            lng = s.get("longitude")
            if lat and lng:
                if is_inside_dasma(float(lat), float(lng))[0]:
                    recent_sites.append((float(lat), float(lng)))

        # 2) Generate candidate points around seeds with randomized patterns
        def meters_to_degrees(m: float) -> float:
            return float(m) / 111000.0

        jitter = meters_to_degrees(radius_meters)
        candidates = []  # (lat, lng, seed_weight)

        # Stronger weight for dengue cases  
        for (lat, lng) in recent_cases:
            points_generated = 0
            max_attempts = max_points_per_seed * 2  # Allow more attempts
            
            for attempt in range(max_attempts):
                if points_generated >= max_points_per_seed:
                    break
                    
                # Random circular distribution instead of fixed grid
                if attempt == 0:
                    # Always include the original point
                    dx, dy = 0, 0
                else:
                    # Random point within radius with some variation
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, jitter) * random.uniform(0.7, 1.3)  # ±30% variation
                    dx = distance * math.cos(angle)
                    dy = distance * math.sin(angle)
                
                lat2 = lat + dy
                lng2 = lng + dx
                
                if is_inside_dasma(lat2, lng2)[0]:
                    candidates.append((lat2, lng2, 1.0))
                    points_generated += 1

        # Moderate weight for known breeding sites
        for (lat, lng) in recent_sites:
            points_generated = 0
            max_attempts = max_points_per_seed
            
            for attempt in range(max_attempts):
                if points_generated >= max_points_per_seed // 2:  # Less points for breeding sites
                    break
                    
                if attempt == 0:
                    # Always include the original point
                    dx, dy = 0, 0
                else:
                    # Smaller radius for breeding sites, more random
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, jitter * 0.8) * random.uniform(0.8, 1.2)
                    dx = distance * math.cos(angle)
                    dy = distance * math.sin(angle)
                
                lat2 = lat + dy
                lng2 = lng + dx
                
                if is_inside_dasma(lat2, lng2)[0]:
                    candidates.append((lat2, lng2, 0.6))
                    points_generated += 1

        # 3) Score each candidate using weather (at point), waterway proximity, and seed weight
        results: List[Dict[str, Any]] = []
        used_keys = set()

        for (lat, lng, seed_w) in candidates:
            key = f"{round(lat,6)}:{round(lng,6)}"
            if key in used_keys:
                continue
            used_keys.add(key)

            try:
                w = await weather_client.get_weather_history(lat, lng)
                rainfall = float(w.get("rainfall", 25.0))
                humidity = float(w.get("humidity", 80.0))
                temperature = float(w.get("temperature", 28.0))
                waterway_distance = float(get_nearest_waterway_distance(lat, lng))

                # Normalize
                norm_rain = max(0.0, min(1.0, rainfall / 60.0))
                norm_hum = max(0.0, min(1.0, humidity / 100.0))
                # Favor ~26–32C
                norm_temp = max(0.0, 1.0 - abs((temperature - 29.0) / 10.0))
                inv_dist = 1.0 / (1.0 + (waterway_distance / 200.0))

                # Combine with seed weight (cases stronger)
                probability = (
                    0.30 * seed_w +
                    0.20 * norm_rain +
                    0.20 * norm_hum +
                    0.10 * norm_temp +
                    0.20 * inv_dist
                )
                probability = float(max(0.0, min(1.0, probability)))
                if probability < min_probability:
                    continue
                
                results.append({
                    "longitude": round(lng, 6),
                    "latitude": round(lat, 6),
                    "risk_level": "high",
                    "probability": round(probability, 3),
                    "predicted_days": predicted_days
                })
            except Exception as e:
                logger.warning(f"Skip candidate {key}: {e}")
                continue
        
        return results
    except Exception as e:
        logger.error(f"Failed to predict potential breeding sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))