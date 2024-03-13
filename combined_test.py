import csv
import gradio as gr
import numpy as np
import pandas as pd
import translator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model

def load_tokenizer_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_string = f.read()
    return tokenizer_from_json(json_string)

sumerian_to_english_model = load_model('Model/sumerian_translation_model.keras')
english_to_sumerian_model = load_model('Model/english_translation_model.keras')
english_tokenizer = load_tokenizer_from_file('Model/english_tokenizer.json')
sumerian_tokenizer = load_tokenizer_from_file('Model/sumerian_tokenizer.json')
pos_tokenizer = load_tokenizer_from_file('Model/pos_tokenizer.json')
csv_file_path='processed_transliteration_data.csv'
df = pd.read_csv(csv_file_path)

def translate_sumerian_to_english(sumerian_input_text):
    translated_words = []

    for word in sumerian_input_text.split():
        if word in df['lemma'].values and df[df['lemma'] == word]['language'].iloc[0] == 'Sumerian':
            word_data = df[(df['lemma'] == word) & (df['language'] == 'Sumerian')].iloc[0]
            pos_tag = word_data['pos']
            category_id = word_data['category']
        else:
            continue
        sumerian_seq = sumerian_tokenizer.texts_to_sequences([word])
        sumerian_padded = pad_sequences(sumerian_seq, maxlen=3, padding='post')
        pos_seq = pos_tokenizer.texts_to_sequences([pos_tag])
        pos_padded = pad_sequences(pos_seq, maxlen=3, padding='post')
        category = np.array([category_id]).reshape(1, -1)
        prediction = sumerian_to_english_model.predict([sumerian_padded, pos_padded, category])
        predicted_index = np.argmax(prediction, axis=-1)[0]
        translated_word = english_tokenizer.index_word.get(predicted_index, "UNKNOWN")
        translated_words.append(translated_word)
    return translator.translate_sumerian_to_english(sumerian_input_text, translated_words)

def translate_english_to_sumerian(english_input_text):
    translated_words = []

    for word in english_input_text.split():
        english_seq = english_tokenizer.texts_to_sequences([word])
        english_padded = pad_sequences(english_seq, maxlen=3, padding='post')

        if word in df['label'].values:
            word_data = df[df['label'] == word].iloc[0]
            pos_tag = word_data['pos']
            category_id = word_data['category']
        else:
            continue
        pos_seq = pos_tokenizer.texts_to_sequences([pos_tag])
        pos_padded = pad_sequences(pos_seq, maxlen=3, padding='post')
        category = np.array([category_id])
        translated_words.append(word)
    return translator.translate_english_to_sumerian(english_input_text, translated_words)

def augment_sentence_with_matching_pair(sentence):
    words = sentence.split()
    word_pairs = [(' '.join(words[i:i + n]), '_'.join(words[i:i + n])) for n in range(2, len(words) + 1) for i in range(len(words) - n + 1)]
    matches = {}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            for original, modified in word_pairs:
                if row['label'] == modified and row['language'] == 'English':
                    matches[original] = modified
    sorted_matches = sorted(matches.items(), key=lambda x: len(x[0]), reverse=True)
    for original, replacement in sorted_matches:
        if original in sentence:
            sentence = sentence.replace(original, replacement)
    return sentence

def translate(text, direction):
    if direction == "English to Sumerian":
        augmented_text = augment_sentence_with_matching_pair(text)
        return translate_english_to_sumerian(augmented_text)
    elif direction == "Sumerian to English":
        return translate_sumerian_to_english(text)

iface = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(lines=5, label="Enter text"),
            gr.Radio(choices=["English to Sumerian", "Sumerian to English"], label="Translation Direction")],
    outputs=gr.Textbox(label="Translated Text"),
    title="Sumerian-English Translation",
    description="Translate text between English and Sumerian",
    allow_flagging="never"
).launch()
