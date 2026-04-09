import sys
import logging
from pathlib import Path

import datafog
from datafog import DataFog


# Add repository root to path to import shared modules
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.getLogger("transformers").setLevel(logging.ERROR)

class PrivacyAgent:
  def __init__(self):
    pass
  
  def preliminary_censor(self,input):
    cleaned_text = datafog.sanitize(input, engine="regex")
    return cleaned_text
  
  def in_depth_censor(self,input):
    cleaned_text = datafog.sanitize(input, engine="smart")
    return cleaned_text