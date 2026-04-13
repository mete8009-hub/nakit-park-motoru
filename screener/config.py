from pathlib import Path

APP_TITLE = "Fund Manager Workstation v3"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "workstation.db"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
NASDAQ_ETF_LIST_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
TEFAS_API_BASE = "https://www.tefas.gov.tr/api/DB"
REPO_INDEX_URL = "https://www.borsaistanbul.com/en/index/repbr"
TPP_URLS = [
    "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-gunluk-bulten",
    "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-islem-ortalamalari-raporu",
]
RATING_SCALE = {"NR":0, "B":1, "BB":2, "BBB":3, "A":4, "AA":5, "AAA":6}
THEME_KEYWORDS = {
    "Green ESG": ["esg", "sustainable", "green", "clean", "climate", "low carbon", "carbon", "solar", "wind", "water"],
    "AI & Technology": ["artificial intelligence", "ai", "robotics", "cloud", "cyber", "semiconductor", "tech", "software"],
    "Dividend": ["dividend", "income"],
    "Gold & Precious Metals": ["gold", "silver", "miners", "precious"],
    "Fixed Income": ["bond", "treasury", "ultra short", "municipal", "high yield", "corp bond", "income", "floating rate"],
    "Crypto": ["bitcoin", "ethereum", "crypto", "blockchain"],
    "EM & Country": ["emerging", "msci", "china", "india", "japan", "europe", "turkey", "asia", "latin"],
}
