import sys
import logging
from pathlib import Path

import datafog
import re


# Add repository root to path to import shared modules
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.getLogger("transformers").setLevel(logging.ERROR)

class PrivacyAgent:
  def __init__(self):
    pass
  
  #Quick method that cleans away dates and SSIDs quickly
  def preliminary_censor(self,text):
    try:
      cleaned_text = datafog.sanitize(text, engine="regex")
      return cleaned_text
    except Exception as e:
      print(f"Error in preliminary_censor: {e}")
      return text 
  
  #More in-depth method that sanitizes names, companies, and more information in addition to dates and SSIDs.
  def in_depth_censor(self,text):
    try:
      cleaned_text = datafog.sanitize(text, engine="smart")
      return cleaned_text
    except Exception as e:
      print(f"Error in in_depth_censor: {e}")
      return text
  
  #Custom censor where the client can choose what words or regex expressions to replace as well as their replacements
  def custom_regex_censor(self,text,bad_words = [],replacements = []):
    if len(bad_words) != len(replacements):
      raise ValueError("bad_words and replacements must be the same length in custom_regex_censor")
    
    for bad_word, replacement in zip(bad_words,replacements):
      text = re.sub(bad_word,replacement,text,flags=re.IGNORECASE)
    
    return text
  
  #Combines the top 3 into one method for ease of use
  def complete_censor(self,text,bad_words=[],replacements=[]):
    text = self.preliminary_censor(text)
    text = self.in_depth_censor(text)
    text = self.custom_regex_censor(text,bad_words,replacements)
    return text
