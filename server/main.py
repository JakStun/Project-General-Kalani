import logging
import logging.config
import yaml

from fastapi import FastAPI, APIRouter
from pathlib import Path

# --- Load logging config ---
LOG_CONFIG_PATH = Path(__file__).parent / "logger_config.yml"

if LOG_CONFIG_PATH.exists():
    with open(LOG_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    # Fallback if config not found
    logging.basicConfig(level=logging.INFO)
    logging.warning("Logging configuration file not found. Using basic config.")

# Create logger for this module
logger = logging.getLogger("main")

app = FastAPI(
    title="Magnificient Lucrehulk",
    description="Brain of all operations, responsible for processing and understanding all data.",
    version="1.0.0",
    docs_url="/api/docs", #TODO: None -> before release
    redoc_url=None,
    openapi_url=f"/api/openapi.json", #TODO: None -> before release
)

router = APIRouter(prefix='/api')

@router.get("/v1/isalive")
async def isalive():
  return {"isalive": True}

@router.get("/v1/hello")
async def hello():
  return {"message": "Hello world"}
