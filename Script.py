import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import cmudict
from nltk import download

# Download required NLTK data files
download('punkt')
download('stopwords')
download('cmudict')

# Load positive and negative words
try:
    with open('positive-words.txt', 'r') as file:
        positive_words = set(file.read().split())
    with open('negative-words.txt', 'r') as file:
        negative_words = set(file.read().split())
except FileNotFoundError as e:
    print(f"Word list file not found: {e}")
    exit()

# Read the input Excel file
try:
    input_df = pd.read_excel('Input.xlsx')
except FileNotFoundError as e:
    print(f"Input file not found: {e}")
    exit()
except Exception as e:
    print(f"Error reading input file: {e}")
    exit()

# Create output directory for text files
if not os.path.exists('articles'):
    os.makedirs('articles')

# Function to count syllables in a word
d = cmudict.dict()

def syllable_count(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        # If word not found in cmudict, assume 1 syllable
        return 1

# Function to extract article text
def extract_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title and article text (adjust the tag extraction according to the website structure)
        title = soup.find('h1').get_text() if soup.find('h1') else ''
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])

        return title + '\n' + article_text
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return ""

# Extract article text from URLs and perform text analysis
output_data = []

for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    print(f"Processing URL ID {url_id}: {url}")

    # Extract the article text
    article_text = extract_article_text(url)

    if not article_text:
        print(f"Skipping URL ID {url_id} due to extraction issues.")
        continue

    # Save the article text to a file
    try:
        with open(f'articles/{url_id}.txt', 'w', encoding='utf-8') as file:
            file.write(article_text)
    except Exception as e:
        print(f"Error saving article text for URL ID {url_id}: {e}")
        continue

    # Perform text analysis
    words = word_tokenize(article_text)
    sentences = sent_tokenize(article_text)

    if len(sentences) == 0:
        print(f"No sentences found in URL ID {url_id}. Skipping.")
        continue

    # Remove stopwords and non-alphabetic tokens for analysis
    words = [word for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]

    positive_score = sum(1 for word in words if word.lower() in positive_words)
    negative_score = sum(1 for word in words if word.lower() in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    avg_sentence_length = len(words) / len(sentences)
    complex_word_count = sum(1 for word in words if syllable_count(word) >= 3)
    percentage_complex_words = (complex_word_count / len(words)) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(words) / len(sentences)
    syllables_per_word = sum(syllable_count(word) for word in words) / len(words)
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', article_text, re.I))
    avg_word_length = sum(len(word) for word in words) / len(words)

    # Append data to output
    output_data.append([
        url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,
        avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence,
        complex_word_count, len(words), syllables_per_word, personal_pronouns, avg_word_length
    ])

    print(f"Finished processing URL ID {url_id}.")

# Create output DataFrame
output_columns = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
    'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
    'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]
output_df = pd.DataFrame(output_data, columns=output_columns)

# Write the output to an Excel file
try:
    output_df.to_excel('Output Data Structure.xlsx', index=False)
    print("All articles processed. Results saved to Output Data Structure.xlsx.")
except Exception as e:
    print(f"Error writing to output file: {e}")
