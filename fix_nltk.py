import nltk

# Force clean download
nltk.download('punkt', download_dir='nltk_data')

# Add data path manually
nltk.data.path.append('./nltk_data')

# Test it
from nltk.tokenize import sent_tokenize

text = "ADAK supports anti-doping. Athletes are protected under WADA code."
print(sent_tokenize(text))

