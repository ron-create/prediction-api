import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        # Supabase configuration
        self.url = os.getenv("SUPABASE_URL", "https://wijncpcaksvtkzhgbcaf.supabase.co")
        self.key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indpam5jcGNha3N2dGt6aGdiY2FmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg2NjI2NzAsImV4cCI6MjA1NDIzODY3MH0.Tx8Ly8Lk6kdbAVlMGVi77viZZPgxzqToXhkw93fk7yE")
        
        # Initialize Supabase client
        self.client = None
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(self.url, self.key)
            logger.info("✅ Supabase client initialized successfully")
        except ImportError as e:
            logger.error(f"❌ Supabase library not installed: {e}")
            logger.error("Run: pip install supabase")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")

    async def fetch_recent_dengue_cases(self, days: int = 30) -> List[Dict]:
        """
        Fetch recent dengue cases across all barangays, including resident latitude/longitude.
        Returns rows containing resident.latitude and resident.longitude when available.
        """
        if not self.client:
            return []
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            query = self.client.table("dengue_cases").select(
                """
                id,
                case_status,
                outcome,
                created_at,
                resident!inner(
                    id,
                    latitude,
                    longitude
                )
                """
            )
            query = query.eq("case_status", "Confirmed")
            query = query.in_("outcome", ["Recovered", "Deceased"])
            query = query.gte("created_at", start_date.isoformat())
            query = query.lte("created_at", end_date.isoformat())
            resp = query.execute()
            return resp.data or []
        except Exception as e:
            logger.error(f"Error fetching recent dengue cases: {e}")
            return []

    async def fetch_recent_breeding_sites(self, days: int = 30) -> List[Dict]:
        """
        Fetch recent breeding sites across all barangays.
        Returns rows with latitude, longitude, created_at.
        """
        if not self.client:
            return []
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            query = self.client.table("breeding_sites_reports").select("id, latitude, longitude, created_at")
            query = query.gte("created_at", start_date.isoformat())
            query = query.lte("created_at", end_date.isoformat())
            resp = query.execute()
            return resp.data or []
        except Exception as e:
            logger.error(f"Error fetching recent breeding sites: {e}")
            return []

# Global instance
supabase_client = SupabaseClient()