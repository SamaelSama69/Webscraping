import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
import os
import re
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

print("Starting Blackcoffer Data Extraction and NLP Analysis...\n")

# --------------------------------
# Paths
# --------------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_PATH, "input", "Input.xlsx")
ARTICLES_FOLDER = os.path.join(BASE_PATH, "articles")
OUTPUT_FILE = os.path.join(BASE_PATH, "output", "Output.xlsx")

POSITIVE_WORDS_FILE = os.path.join(BASE_PATH, "dictionaries", "positive-words.txt")
NEGATIVE_WORDS_FILE = os.path.join(BASE_PATH, "dictionaries", "negative-words.txt")

STOPWORDS_FOLDER = os.path.join(BASE_PATH, "stopwords")

os.makedirs(ARTICLES_FOLDER, exist_ok=True)

# --------------------------------
# HTTP Session
# --------------------------------

session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9"
})

# --------------------------------
# Load Sentiment Dictionaries
# --------------------------------

def load_words(file):

    words = set()

    with open(file, 'r', encoding='latin-1') as f:

        for line in f:

            line = line.strip()

            if line.startswith(";") or line == "":
                continue

            words.add(line.lower())

    return words


positive_words = load_words(POSITIVE_WORDS_FILE)
negative_words = load_words(NEGATIVE_WORDS_FILE)

print("Positive words:", len(positive_words))
print("Negative words:", len(negative_words))

# --------------------------------
# Load Stopwords
# --------------------------------

def load_stopwords(folder):

    stopwords = set()

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        with open(path, 'r', encoding='latin-1') as f:

            for line in f:

                word = line.strip().split('|')[0].strip().lower()

                if word:
                    stopwords.add(word)

    return stopwords


stop_words = load_stopwords(STOPWORDS_FOLDER)

print("Stopwords:", len(stop_words))

# --------------------------------
# Extract Article
# --------------------------------

def extract_article(url):

    for attempt in range(3):

        try:

            response = session.get(url, timeout=15)

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "lxml")

            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else ""

            # detect article container
            article = soup.find("div", class_="td-post-content")

            if article is None:
                article = soup.find("div", class_="tdb-block-inner")

            if article is None:
                article = soup.find("div", class_="entry-content")

            if article is None:
                article = soup.find("article")

            if article is None:
                article = soup.select_one("div[class*='content']")

            # fallback: get all paragraphs
            if article is None:
                paragraphs = soup.find_all("p")
            else:
                paragraphs = article.find_all("p")

            text = " ".join(p.get_text(strip=True) for p in paragraphs)

            # remove URLs from article text
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)

            if len(text) > 200:
                return title, text

        except:
            pass

    return "", ""

# --------------------------------
# Syllable Counter
# --------------------------------

def syllable_count(word):

    vowels = "aeiou"

    count = 0

    for i in range(len(word)):

        if word[i] in vowels:

            if i == 0 or word[i-1] not in vowels:
                count += 1

    if word.endswith("es") or word.endswith("ed"):
        count -= 1

    if count <= 0:
        count = 1

    return count

# --------------------------------
# Pronoun Counter
# --------------------------------

def count_pronouns(text):

    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.I)

    return len(pronouns)

# --------------------------------
# Text Analysis
# --------------------------------

def analyze(text):

    # remove URLs again for safety
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    sentences = sent_tokenize(text)

    tokens = word_tokenize(text.lower())

    tokens = [re.sub(r'[^a-zA-Z]', '', w) for w in tokens]
    tokens = [w for w in tokens if w]

    # sentiment words keep stopwords
    sentiment_words = tokens

    # readability words remove stopwords
    words = [w for w in tokens if w not in stop_words]

    word_count = len(words)

    if word_count == 0:
        return [0]*13

    positive_score = sum(1 for w in sentiment_words if w in positive_words)
    negative_score = sum(1 for w in sentiment_words if w in negative_words)

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

    subjectivity_score = (positive_score + negative_score) / (word_count + 0.000001)

    avg_sentence_length = word_count / (len(sentences) + 1)

    avg_words_per_sentence = avg_sentence_length

    complex_words = [w for w in words if syllable_count(w) > 2]

    complex_word_count = len(complex_words)

    percentage_complex_words = complex_word_count / word_count

    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    syllables = sum(syllable_count(w) for w in words)

    syllables_per_word = syllables / word_count

    pronouns = count_pronouns(text)

    avg_word_length = sum(len(w) for w in words) / word_count

    return [
        positive_score,
        negative_score,
        polarity_score,
        subjectivity_score,
        avg_sentence_length,
        percentage_complex_words,
        fog_index,
        avg_words_per_sentence,
        complex_word_count,
        word_count,
        syllables_per_word,
        pronouns,
        avg_word_length
    ]

# --------------------------------
# Process Each Row
# --------------------------------

def process_row(row):

    url_id = row["URL_ID"]
    url = row["URL"]

    print("Processing:", url)

    title, text = extract_article(url)

    file_path = os.path.join(ARTICLES_FOLDER, f"{url_id}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(text)

    metrics = analyze(text)

    return [url_id, url] + metrics

# --------------------------------
# Main Program
# --------------------------------

df = pd.read_excel(INPUT_FILE)

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

columns = [
    "URL_ID",
    "URL",
    "POSITIVE SCORE",
    "NEGATIVE SCORE",
    "POLARITY SCORE",
    "SUBJECTIVITY SCORE",
    "AVG SENTENCE LENGTH",
    "PERCENTAGE OF COMPLEX WORDS",
    "FOG INDEX",
    "AVG NUMBER OF WORDS PER SENTENCE",
    "COMPLEX WORD COUNT",
    "WORD COUNT",
    "SYLLABLE PER WORD",
    "PERSONAL PRONOUNS",
    "AVG WORD LENGTH"
]

output_df = pd.DataFrame(results, columns=columns)

output_df.to_excel(OUTPUT_FILE, index=False)

print("\nAnalysis Completed!")
print("Output saved to:", OUTPUT_FILE)