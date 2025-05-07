import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files (stopwords, WordNet) if not already present
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize stopwords set and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Clean and normalize a text string:
      - Lowercase
      - Remove punctuation/non-word characters
      - Tokenize
      - Remove stopwords
      - Lemmatize tokens
    Returns the cleaned text as a single string.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and non-word characters (keep spaces)
    text = re.sub(r'\W+', ' ', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    # Remove stopwords and lemmatize each token
    tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in stop_words
    ]
    # Join tokens back into a single string
    return ' '.join(tokens)

def add_numeric_features(df):
    """
    Add custom numeric features to the DataFrame:
      - word_count: number of words in the original title
      - exclamation_count: number of '!' characters
      - upper_case_count: number of words that are fully uppercase
    Modifies the DataFrame in place and returns it.
    """
    # Word count (by splitting on whitespace)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    # Count of exclamation marks
    df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
    # Count of fully uppercase words (e.g., 'BREAKING')
    df['upper_case_count'] = df['text'].apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper())
    )
    return df
