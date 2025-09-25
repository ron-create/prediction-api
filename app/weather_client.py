import httpx
import logging
from typing import Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WeatherClient:
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    async def get_weather_history(self, lat: float, lng: float) -> Dict[str, float]:
        """
        Get 14-day weather history from Open-Meteo (completely free)
        Returns average rainfall, humidity, and temperature over the period.
        """
        try:
            # Get 14-day historical data from same period last year
            today = datetime.now()
            end_date = today.replace(year=today.year - 1) - timedelta(days=1)
            start_date = end_date - timedelta(days=13)
            
            params = {
                "latitude": lat,
                "longitude": lng,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "daily": "precipitation_sum,relative_humidity_2m_mean,temperature_2m_mean",
                "timezone": "Asia/Manila"
            }
            
            logger.info(f"Requesting Open-Meteo data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} for ({lat}, {lng})")
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                daily = data.get("daily", {})
                
                # Extract data arrays
                rainfall_data = daily.get("precipitation_sum", [])
                humidity_data = daily.get("relative_humidity_2m_mean", [])
                temp_data = daily.get("temperature_2m_mean", [])
                
                # Filter out None values and calculate averages
                valid_rainfall = [r for r in rainfall_data if r is not None]
                valid_humidity = [h for h in humidity_data if h is not None]
                valid_temp = [t for t in temp_data if t is not None]
                
                avg_rainfall = sum(valid_rainfall) / len(valid_rainfall) if valid_rainfall else 25.0
                avg_humidity = sum(valid_humidity) / len(valid_humidity) if valid_humidity else 80.0
                avg_temperature = sum(valid_temp) / len(valid_temp) if valid_temp else 28.0
                
                weather_data = {
                    "rainfall": round(avg_rainfall, 2),
                    "humidity": round(avg_humidity, 2),
                    "temperature": round(avg_temperature, 2)
                }
                
                logger.info(f"Open-Meteo historical data fetched for ({lat}, {lng}): {weather_data}")
                return weather_data
                
        except httpx.TimeoutException:
            logger.warning(f"Open-Meteo API timeout for ({lat}, {lng}), using defaults")
            return {"rainfall": 25.0, "humidity": 80.0, "temperature": 28.0}
        except httpx.HTTPStatusError as e:
            logger.warning(f"Open-Meteo API HTTP error {e.response.status_code} for ({lat}, {lng}), using defaults")
            return {"rainfall": 25.0, "humidity": 80.0, "temperature": 28.0}
        except KeyError as e:
            logger.warning(f"Open-Meteo API missing data field {e} for ({lat}, {lng}), using defaults")
            return {"rainfall": 25.0, "humidity": 80.0, "temperature": 28.0}
        except Exception as e:
            logger.error(f"Open-Meteo API error for ({lat}, {lng}): {e}")
            return {"rainfall": 25.0, "humidity": 80.0, "temperature": 28.0}

# Global instance
weather_client = WeatherClient()    