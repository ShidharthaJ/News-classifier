import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# keep a few words that carry signal in headlines
STOP = set(ENGLISH_STOP_WORDS) - {"us", "first", "new"}

def clean_text(text: str) -> str:
    """Lowercase -> strip non-letters -> tokenize -> drop stopwords/short tokens."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in STOP]
    return " ".join(tokens)

# Raw TOI category prefix -> clean target class
CATEGORY_MAP = {
    "city": "City / Local",
    "india": "India / National",
    "entertainment": "Entertainment",
    "tv": "Entertainment",
    "business": "Business",
    "sports": "Sports",
    "cricket": "Sports",
    "world": "World",
    "life-style": "Lifestyle",
    "tech": "Technology",
}
