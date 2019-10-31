from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pickle
import nltk
import re

nltk.download('stopwords')

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

# Function courtesy: http://bit.ly/2N17Clt
def clean_title(title: np.ndarray) -> np.ndarray:
    # lower case and remove special characters\whitespaces
    title = re.sub(r'[^a-zA-Z\s]', '', title, re.I|re.A)
    title = title.lower()
    title = title.strip()
    # tokenize document
    tokens = wpt.tokenize(title)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    title = ' '.join(filtered_tokens)
    return title
    