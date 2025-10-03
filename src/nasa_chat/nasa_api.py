import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


class NASAAPI:
    """
    Minimal client for common NASA public APIs.
    Uses NASA_API_KEY from environment, falls back to DEMO_KEY.
    """

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.session = session or requests.Session()

    # ---------- Planetary: APOD ----------
    def get_apod(self, date: Optional[str] = None, thumbs: bool = True) -> Dict[str, Any]:
        url = "https://api.nasa.gov/planetary/apod"
        params = {"api_key": self.api_key, "thumbs": "true" if thumbs else "false"}
        if date:
            params["date"] = date
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    # ---------- Mars Rover Photos ----------
    def get_mars_rover_photos(
        self,
        rover: str = "curiosity",
        sol: Optional[int] = None,
        earth_date: Optional[str] = None,
        camera: Optional[str] = None,
        page: int = 1,
    ) -> Dict[str, Any]:
        """
        Retrieve photos from a Mars rover.
        Provide either sol (Martian day) or earth_date (YYYY-MM-DD).
        """
        base = f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
        params: Dict[str, Any] = {"api_key": self.api_key, "page": page}
        if sol is not None:
            params["sol"] = sol
        if earth_date is not None:
            params["earth_date"] = earth_date
        if camera:
            params["camera"] = camera
        r = self.session.get(base, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    # ---------- EPIC (Earth) ----------
    def get_epic_images(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        EPIC images metadata.
        If date is None, returns most recent images metadata list.
        """
        base = "https://api.nasa.gov/EPIC/api/natural"
        url = f"{base}/images" if date is None else f"{base}/date/{date}"
        params = {"api_key": self.api_key}
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data:
            # Construct PNG archive URL for convenience
            dt = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")
            y, m, d = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
            image = item["image"]
            item["image_url_png"] = f"https://epic.gsfc.nasa.gov/archive/natural/{y}/{m}/{d}/png/{image}.png"
        return data

    # ---------- NASA Image and Video Library ----------
    def search_images(self, query: str, media_type: str = "image", page: int = 1) -> List[Dict[str, Any]]:
        """
        Search the NASA Image and Video Library for images by keyword.
        Returns simplified records with title, description, preview_url, nasa_id, and links.
        """
        base = "https://images-api.nasa.gov/search"
        params = {"q": query, "media_type": media_type, "page": page}
        r = self.session.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("collection", {}).get("items", [])
        results: List[Dict[str, Any]] = []
        for it in items:
            data_block = (it.get("data") or [{}])[0]
            links = it.get("links") or []
            preview_url = None
            for ln in links:
                if ln.get("rel") == "preview" or ln.get("render") == "image":
                    preview_url = ln.get("href")
                    break
            results.append(
                {
                    "title": data_block.get("title"),
                    "description": data_block.get("description"),
                    "nasa_id": data_block.get("nasa_id"),
                    "center": data_block.get("center"),
                    "keywords": data_block.get("keywords", []),
                    "date_created": data_block.get("date_created"),
                    "preview_url": preview_url,
                    "links": links,
                    "href": it.get("href"),  # JSON list of asset links
                }
            )
        return results

    def get_asset_urls(self, href_url: str) -> List[str]:
        """
        Given an 'href' from search results, return list of asset URLs (images of various sizes).
        """
        r = self.session.get(href_url, timeout=30)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            # Some endpoints return plain text with newline-separated URLs
            text = r.text.strip()
            return [line for line in text.splitlines() if line.strip()]