import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "trafficSight.log"
LINE_CONFIG_FILE = BASE_DIR / "counting_lines.json"
GEOSPATIAL_CALIB_FILE = BASE_DIR / "geospatial_calibration.json"

DEFAULT_DATABASE_URL = "postgresql://trafficsight:trafficsight@localhost:5432/trafficsight"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

STREAM_URLS = {
    "Sugeng Jeroni 2": "http://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SugengJeroni2.stream/playlist.m3u8",
    "Simpang Wirosaban Barat": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Simpang_Wirosaban_View_Barat.stream/playlist.m3u8",
    "Simpang Pingit": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_Pingit2.stream/playlist.m3u8",
}
DEFAULT_STREAM_NAME = "Sugeng Jeroni 2"
DEFAULT_STREAM_URL = STREAM_URLS[DEFAULT_STREAM_NAME]

WIDTH = 1920
HEIGHT = 1080
BUFFER_SECONDS = 60
FALLBACK_FPS = 25.0
OVERSPEED_KMH = 60.0
SPEED_CAP_KMH = 140.0
LINES_FILE = LINE_CONFIG_FILE
IS_EDITING_LINES = False

DEFAULT_COUNTING_LINES = {
    "Utara":  {"type": "H", "y": 310, "x1": 350, "x2": 700},
    "Selatan":{"type": "H", "y": 580, "x1": 350, "x2": 750},
    "Barat":  {"type": "V", "x": 150, "y1": 310, "y2": 580},
    "Timur":  {"type": "V", "x": 970, "y1": 260, "y2": 520},
}
