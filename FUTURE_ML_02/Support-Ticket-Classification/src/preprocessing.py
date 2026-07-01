import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import pandas as pd

# Download necessary NLTK data quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def expand_contractions(text: str) -> str:
    """Expands contractions in the text (e.g., don't -> do not)."""
    return contractions.fix(text)

def remove_urls(text: str) -> str:
    """Removes URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_html_tags(text: str) -> str:
    """Removes HTML tags from the text."""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def remove_special_characters_and_numbers(text: str) -> str:
    """Removes special characters and numbers, keeping only alphabets and spaces."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def normalize_whitespace_and_case(text: str) -> str:
    """Removes extra whitespace and converts to lowercase."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> str:
    """Tokenizes text, removes stopwords, and lemmatizes tokens."""
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS]
    return ' '.join(tokens)

def preprocess_text(text: str) -> str:
    """
    Applies the full preprocessing pipeline to a single text string.
    """
    if not isinstance(text, str):
        return ""
    
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_special_characters_and_numbers(text)
    text = normalize_whitespace_and_case(text)
    text = tokenize_and_lemmatize(text)
    
    return text

def preprocess_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Applies preprocessing to a specific column in a pandas DataFrame.
    """
    df = df.copy()
    df['cleaned_text'] = df[column_name].apply(preprocess_text)
    return df