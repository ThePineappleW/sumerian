import csv
from pathlib import Path
import pandas as pd
import nltk
from nltk.corpus import words, stopwords
import spacy

from Data.data_utils import load_json, process_file, skip_row_check, clean_label, categorize_label, normalize_label, \
    modify_specific_entries, english_value_check, split_labels_and_duplicate_rows, load_word_vectors, preprocess_labels, \
    apply_pca_and_clustering, load_category_lookup, map_sumerian_to_categories, conjugate_to_third_person, \
    augment_conjugation

nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
english_words = set(words.words())
stop_words = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_sm')
special_terms_path = Path("special_terms.json")
data = load_json(special_terms_path)
custom_terms = {term.lower() for term in data['special_terms']}
remove_labels = {term.lower() for term in data['special_removal_labels']}


def extract_transliteration_data(input_dir, output_file):
    seen_entries = set()
    input_transliteration_data = Path(input_dir)
    output_transliteration_data = Path(output_file)

    with output_transliteration_data.open('w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'n', 'id', 'corresp', 'form', 'lemma', 'pos', 'label', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for xml_file in input_transliteration_data.glob('*.xml'):
            lines_info = process_file(xml_file)

            for line in lines_info:
                for word in line['words']:
                    row = {
                        'file_name': xml_file.name,
                        'n': line['n'],
                        'id': line['id'],
                        'corresp': line['corresp'],
                        **word
                    }

                    if skip_row_check(row):
                        continue

                    row['label'] = clean_label(row['label'])
                    row['category'] = categorize_label(row['label']) if row['pos'].upper() == 'N' else "N/A"

                    custom_key = (row['n'], row['id'], row['form'], row['lemma'], row['pos'], row['label'], row['category'])

                    if custom_key not in seen_entries:
                        seen_entries.add(custom_key)
                        writer.writerow(row)


def process_and_normalize_csv_data(input_output_cleaned_data, output_output_cleaned_data):
    seen = set()
    pre_filtered_rows = []
    fieldnames = ['file_name', 'n', 'id', 'corresp', 'form', 'lemma', 'pos', 'label', 'category']

    with open(input_output_cleaned_data, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames.append('Language')

        for row in reader:
            row['label'] = normalize_label(row['label']).lower()
            row['label'] = modify_specific_entries(row['label'], row['pos'])

            if row.get('category') == 'N/A':
                if row['pos'] == 'N':
                    row['Language'] = 'English' if english_value_check(row['label'], english_words) else 'Sumerian'
                else:
                    row['Language'] = 'Not Applicable'
            else:
                row['Language'] = 'Sumerian'

            if row['pos'] == 'V' and not row['label'].startswith("to"):
                continue

            if row['pos'] == 'PD' and ',' in row['label']:
                split_rows = split_labels_and_duplicate_rows(row)
                for split_row in split_rows:
                    if split_row.get('category') == 'N/A':
                        split_row['Language'] = 'English' if english_value_check(split_row['label'], english_words) else 'Sumerian'
                    else:
                        split_row['Language'] = 'Sumerian'
                pre_filtered_rows.extend(split_rows)
            else:
                pre_filtered_rows.append(row)

            if row['Language'] == 'Not Available':
                row['Language'] = 'English'

        unique_rows = [row for row in pre_filtered_rows if
                       (row['form'], row['lemma'], row['pos'], row['category'])
                       not in seen and
                       not seen.add((row['form'], row['lemma'], row['pos'], row['category']))]

        filtered_rows = [row for row in unique_rows if not (row['Language'] == 'Sumerian' and row['category'] == 'N/A')]

        final_rows = [row for row in filtered_rows if row.get('label') not in remove_labels]

        final_rows = [dict(row, Language='English') if row.get('Language') == 'Not Applicable' else row for row in final_rows]

        final_rows = [
            row for row in final_rows
            if row.get('pos') not in ['NU', 'I', 'NEG'] and
               not (row.get('pos') == 'AV' and row.get('label').startswith("type of")) and
               not (row.get('pos') == 'N' and row.get('label').startswith("type of")) and
               not (row.get('pos') == 'PD' and '.' in row.get('label')) and
               row.get('label').strip() != ''
        ]

    with open(output_output_cleaned_data, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)


def categorize_and_cluster_transliteration_data(model_path, file_path, category_file_path, output_file_path):
    word_vectors_model = load_word_vectors(model_path)
    df = pd.read_csv(file_path)

    df_english = preprocess_labels(df, 'English')
    df_english = apply_pca_and_clustering(df_english, word_vectors_model)
    max_english_cluster_index = df_english['cluster'].max()

    category_lookup = load_category_lookup(category_file_path)

    df_sumerian = preprocess_labels(df, 'Sumerian')
    df_sumerian = map_sumerian_to_categories(df_sumerian, category_lookup, max_english_cluster_index)

    df_combined = pd.concat([df_english, df_sumerian])
    df_combined.drop(columns=['category'], inplace=True)
    df_combined['category'] = df_combined['cluster']
    df_combined.drop(columns=['label_clean', 'vector', 'cluster'], inplace=True)
    df_combined.to_csv(output_file_path, index=False)


def normalize_and_update_transliteration_data(output_cleaned_data):
    df = pd.read_csv(output_cleaned_data)

    verbs_df = df[df['pos'] == 'V'].copy()
    verbs_df['conjugated'] = conjugate_to_third_person(verbs_df['label'].tolist())
    verbs_df['label'] = verbs_df.apply(augment_conjugation, axis=1)
    verbs_df['label'] = verbs_df['label'].apply(lambda x: '_'.join(x.split()) if ' ' in x else x)

    df.update(verbs_df[['label']])

    non_verbs_df = df[df['pos'] != 'V']
    non_verbs_df['label'] = non_verbs_df['label'].apply(lambda x: '_'.join(x.split()) if ' ' in x else x)
    df.update(non_verbs_df[['label']])

    df.to_csv(output_cleaned_data, index=False)

    df = pd.read_csv(output_cleaned_data)

    df['form'] = df['form'].str.lower()
    df['lemma'] = df['lemma'].str.lower()

    df['form'] = df['form'].str.replace('\d+', '', regex=True)
    df['lemma'] = df['lemma'].str.replace('\d+', '', regex=True)

    df = df[~df['form'].str.contains('x-|-\w*x', regex=True)]

    df = df[df['pos'] != 'PD']

    df.to_csv(output_cleaned_data, index=False)


input_transliteration_dir = Path("ETCSL/transliterations")
output_cleaned_data = "consolidated_transliteration_data.csv"

extract_transliteration_data(input_transliteration_dir, output_cleaned_data)
process_and_normalize_csv_data(output_cleaned_data, output_cleaned_data)
categorize_and_cluster_transliteration_data('GoogleNews-vectors-negative300.bin', output_cleaned_data,'categories.json', output_cleaned_data)
normalize_and_update_transliteration_data(output_cleaned_data)
