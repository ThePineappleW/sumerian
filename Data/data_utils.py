# Standard library
import re
import json
from pathlib import Path

# NLP and Text handling
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import words, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import spacy
from pyinflect import getInflection
from langdetect import detect, LangDetectException
from spellchecker import SpellChecker

# Data handling
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from lxml import etree

# NLTK resource downloads
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

english_words = set(words.words())
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def categorize_label(label):
    categories_path = Path("categories.json")
    categories_data = load_json(categories_path)
    categories = {category: set(keywords) for category, keywords in categories_data.items()}

    label_lower = label.lower()
    for category, keywords in categories.items():
        if label_lower in keywords:
            return category
    return "N/A"


def english_token_check(word):
    return word.lower() in english_words and word.lower() not in stop_words


def english_token_check_2(label):
    return len(label) > 2 and not label.isupper() and not any(char.isdigit() or char == '+' for char in label)


def is_known_meaning(label):
    return "meaning unknown" not in label.lower()


def clean_label(label):
    return label.replace('?', '')


def plus_sign_check(row):
    return any('+' in value for value in row.values())


def valid_verb_check(label, pos):
    if pos != 'V':
        return True
    tokens = word_tokenize(label)
    tagged = pos_tag(tokens)
    return any(tag.startswith('V') for word, tag in tagged) and is_known_meaning(label)


def extract_word_info(word):
    return {
        "form": word.get("form", "N/A"),
        "lemma": word.get("lemma", "N/A"),
        "pos": word.get("pos", "N/A"),
        "label": word.get("label", "N/A")
    }


def process_line(line):
    words_info = [extract_word_info(w) for w in line.findall('w')]
    return {
        "n": line.get("n", "N/A"),
        "id": line.get("id", "N/A"),
        "corresp": line.get("corresp", "N/A"),
        "words": words_info
    }


def process_file(xml_file):
    parser = etree.XMLParser(recover=True, no_network=False)
    tree = etree.parse(str(xml_file), parser)
    return [process_line(line) for line in tree.findall('.//l')]


def skip_row_check(row):
    if any(row.get(field, 'N/A') == 'X' for field in ['form', 'lemma', 'pos', 'label']) or row.get('pos', 'N/A') in [
        'N/A', 'X']:
        return True

    label_lower = row.get('label', '').lower()

    return 'meaning unknown' in label_lower or label_lower == 'x' or (
                row.get('pos', '').upper() == 'V' and label_lower == row.get('lemma', '').lower())


def normalize_label(label):
    normalized_label = re.sub(r'\(([^)]+)\)', r'\1', label)
    normalized_label = re.sub(r'\s{2,}', ' ', normalized_label).strip()
    normalized_label = normalized_label.replace("'s", "")
    return normalized_label


def split_labels_and_duplicate_rows(row, label_field='label'):
    labels = [label.strip() for label in row[label_field].split(',')]
    new_rows = [row.copy() for _ in labels]
    for new_row, label in zip(new_rows, labels):
        new_row[label_field] = label
    return new_rows


def modify_specific_entries(label, pos_tag):
    if pos_tag == 'PD' and label == 'you sg.':
        return 'you'
    return label


special_terms_path = Path("special_terms.json")
data = load_json(special_terms_path)
custom_terms = {term.lower() for term in data['special_terms']}
remove_labels = {term.lower() for term in data['special_removal_labels']}


def english_value_check(label, word_set):
    label_lower = label.lower()

    if label_lower in custom_terms:
        return True

    label_normalized = re.sub(r'[^\w\s]', '', label_lower)
    words_in_label = label_normalized.split()
    if all(word in word_set for word in words_in_label):
        return True

    try:
        if detect(label) == 'en':
            return True
    except LangDetectException:
        pass

    return False


lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def get_wordnet_pos(treebank_tag):
    if treebank_tag == 'AJ':
        return 'a'  # adjective
    elif treebank_tag == 'V':
        return 'v'  # verb
    elif treebank_tag == 'N':
        return 'n'  # noun
    elif treebank_tag == 'AV':
        return 'r'  # adverb
    else:
        return None  # For POS tags with no direct mapping


def correct_and_standardize_translation(row):
    if 'category' not in row or 'label' not in row:
        return row

    if row['category'] != 'English' or pd.isna(row['label']):
        return row

    label = str(row['label']).lower().strip()

    if label:
        spell = SpellChecker()
        tokens = word_tokenize(label)
        corrected_tokens = [spell.correction(token) for token in tokens]

        lemmatized_tokens = corrected_tokens
        if 'pos' in row and row['pos']:
            pos = row['pos']
            wn_pos = get_wordnet_pos(pos.upper())
            if wn_pos:
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(token, pos=wn_pos) for token in corrected_tokens]

        row['label'] = ' '.join(lemmatized_tokens)
    else:
        row['label'] = ''

    return row


def load_word_vectors(model_path):
    return KeyedVectors.load_word2vec_format(model_path, binary=True)


def text_to_vector(text, model):
    words = [subword for word in text.split() for subword in word.split('_')]
    words = [word for word in words if word in model.key_to_index]
    if not words:
        return np.zeros(model.vector_size)
    word_vectors = [model[word] for word in words]
    return np.mean(word_vectors, axis=0)


def preprocess_labels(df, language):
    df_lang = df[df['Language'] == language].copy()
    df_lang['label_clean'] = df_lang['label'].str.lower().str.replace('[^\w\s]', '', regex=True)
    return df_lang


def apply_pca_and_clustering(df, model):
    df['vector'] = df['label_clean'].apply(lambda x: text_to_vector(x, model))
    pca = PCA(n_components=0.95)
    reduced_vectors = pca.fit_transform(np.vstack(df['vector'].values))
    clustering = AgglomerativeClustering(n_clusters=135, affinity='euclidean', linkage='ward')
    df['cluster'] = clustering.fit_predict(reduced_vectors)
    return df


def load_category_lookup(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def determine_cluster(word, category_lookup, category_to_id_map):
    normalized_word = word.replace('_', '-')
    for category, words in category_lookup.items():
        if normalized_word in words:
            return category_to_id_map[category]
    return -1


def map_sumerian_to_categories(df, category_lookup, max_index):
    category_to_id_map = {category: idx + max_index + 1 for idx, category in enumerate(category_lookup.keys())}
    df['cluster'] = df['label'].apply(lambda word: determine_cluster(word, category_lookup, category_to_id_map))
    return df


def conjugate_to_third_person(labels):
    conjugated_labels = []
    for label in labels:
        # Normalize the label by replacing underscores with spaces (if necessary)
        normalized_label = label.replace('_', ' ')
        doc = nlp(normalized_label)
        verb = None
        remainder = []
        for token in doc:
            if token.text.lower() == 'to':
                continue
            if token.pos_ == "AUX" or token.pos_ == "VERB":
                if not verb:
                    verb = token
            else:
                remainder.append(token.text)
        if verb:
            conjugated_verb = getInflection(verb.lemma_, 'VBZ')[0] if getInflection(verb.lemma_, 'VBZ') else verb.text
            conjugated_label = f"{conjugated_verb} {' '.join(remainder)}"
        else:
            conjugated_label = ' '.join(remainder)
        conjugated_labels.append(conjugated_label)
    return conjugated_labels


def augment_conjugation(row):
    conjugated_value = row['conjugated'].strip()
    if conjugated_value == 'is':
        terms = row['label'].split('_')
        if len(terms) > 2:
            return 'is ' + ' '.join(terms[2:])
        else:
            return 'is'
    else:
        return row['conjugated']
