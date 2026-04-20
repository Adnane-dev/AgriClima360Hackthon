# =============================================================
# CONFIGURATION AGRIclima360
# =============================================================

# --- Clés API (à remplacer par vos clés réelles) ---
NOAA_TOKEN = "oAlEkhGLpUtHCIGoUOepslRpcWmtLJMM"  # Token de démonstration
# Obtenez votre token gratuit sur : https://www.ncdc.noaa.gov/cdo-web/token

# --- Paramètres Tunisie ---
TUNISIA_STATIONS = {
    "Bizerte (Nord)":        {"id": "GHCND:TN000060790", "lat": 37.25, "lon": 9.87,  "region": "Nord"},
    "Béja (Nord)":           {"id": "GHCND:TN000060590", "lat": 36.73, "lon": 9.18,  "region": "Nord"},
    "Jendouba (Nord)":       {"id": "GHCND:TN000060490", "lat": 36.50, "lon": 8.78,  "region": "Nord"},
    "Tunis-Carthage (Nord)": {"id": "GHCND:TN000060680", "lat": 36.83, "lon": 10.23, "region": "Nord"},
    "Kairouan (Centre)":     {"id": "GHCND:TN000060750", "lat": 35.67, "lon": 10.10, "region": "Centre"},
    "Kasserine (Centre)":    {"id": "GHCND:TN000060540", "lat": 35.17, "lon": 8.83,  "region": "Centre"},
    "Sfax (Centre)":         {"id": "GHCND:TN000060830", "lat": 34.72, "lon": 10.69, "region": "Centre"},
    "Gabès (Sud)":           {"id": "GHCND:TN000060870", "lat": 33.88, "lon": 10.10, "region": "Sud"},
    "Médenine (Sud)":        {"id": "GHCND:TN000060900", "lat": 33.35, "lon": 10.50, "region": "Sud"},
    "Tataouine (Sud)":       {"id": "GHCND:TN000060950", "lat": 32.93, "lon": 10.45, "region": "Sud"},
}

FAO_CROPS = {
    "Blé dur":    {"item_code": "15",  "element_code": "5419"},
    "Blé tendre": {"item_code": "16",  "element_code": "5419"},
    "Orge":       {"item_code": "44",  "element_code": "5419"},
    "Maïs":       {"item_code": "56",  "element_code": "5419"},
    "Tomate":     {"item_code": "388", "element_code": "5419"},
    "Olive":      {"item_code": "249", "element_code": "5419"},
}

FAO_AREA_TUNISIA = "212"

# --- Paramètres par défaut ---
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2024
DEFAULT_CROP = "Blé dur"
DEFAULT_REGION_PRCP = 500  # mm/an