import csv
import re
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example, offsets_to_biluo_tags
import random

csv_path = Path("../../consolidated_translation_data.csv")

entity_map = {
    "DN": "DEITY",
    "EN": "ETHNONYM",
    "GN": "GEOGRAPHIC",
    "MN": "MONTH",
    "ON": "OBJECT",
    "PN": "PERSON",
    "RN": "ROYAL",
    "SN": "SETTLEMENT",
    "TN": "TOPONYM",
    "WN": "WATERCOURSE"
}

def extract_entities_and_adjust_indices(annotated_text):
    pattern = re.compile(r'\[(DN|EN|GN|MN|ON|PN|RN|SN|TN|WN): ([^\]]+)\]')
    entities = []
    offset = 0
    cleaned_text = annotated_text

    for match in pattern.finditer(annotated_text):
        entity_type_abbr, entity_name = match.groups()
        entity_type = entity_map.get(entity_type_abbr, "UNKNOWN")

        start_index = match.start() - offset
        end_index = start_index + len(entity_name)

        entities.append((start_index, end_index, entity_type))
        full_match_len = len(match.group(0))
        entity_len = len(entity_name)

        offset += full_match_len - entity_len

        cleaned_text = cleaned_text[:start_index] + entity_name + cleaned_text[start_index + full_match_len:]

    return cleaned_text, {'entities': entities}

TRAIN_DATA = []
with csv_path.open(mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        original_text = row['text']
        cleaned_text, entity_info = extract_entities_and_adjust_indices(original_text)

        TRAIN_DATA.append((cleaned_text, entity_info))

for cleaned_text, entities in TRAIN_DATA:
    print("Cleaned Text:", cleaned_text)
    print("Entities:", entities)
    print("\n---\n")

nlp = spacy.blank('xx')

if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)

entity_labels = [
    "DEITY", "ETHNONYM", "GEOGRAPHIC", "MONTH",
    "OBJECT", "PERSON", "ROYAL", "SETTLEMENT", "TOPONYM", "WATERCOURSE"
]
for label in entity_labels:
    ner.add_label(label)

optimizer = nlp.initialize()

for iteration in range(100):
    random.shuffle(TRAIN_DATA)
    losses = {}

    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            biluo_tags = offsets_to_biluo_tags(doc, annotations['entities'])

            if '-' in biluo_tags:
                print(f"Misaligned entities in text: '{text}'")
                for i, tag in enumerate(biluo_tags):
                    if tag == '-':
                        token = doc[i]
                        print(f"  Misaligned token: '{token.text}' at position {token.idx}")
            else:
                examples.append(example)

        nlp.update(examples, sgd=optimizer, drop=0.5, losses=losses)

    print(f"Iteration {iteration}, Losses: {losses}")

model_dir = "ner_1"
nlp.to_disk(model_dir)
print(f"Model saved to {model_dir}")
