import csv
import re
from collections import defaultdict
from lxml import etree
from pathlib import Path

xml_directory = Path("Data/ETCSL/translations")
output_cleaned_data = Path("consolidated_translation_data.csv")
temp_csv = Path("temp_consolidated_translation_data.csv")


def clean_text(text):
    text.replace('"', '').replace('"', '').replace('"', '').replace(',', '')
    text = re.sub(r'--+', ' ', text)  # Replace sequences of dashes with a space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_w_tag(w):
    # Extract the deity or entity name and its type from <w> tag
    name = w.text.strip() if w.text else ""
    type = w.get("type", "unknown")
    # Return a structured representation, encapsulating the type and name
    return f"[{type}: {name}]"


def process_p_tag(p):
    paragraph_id = p.get("id", "N/A")
    corresp = p.get("corresp", "N/A")

    processed_text = []

    # Add paragraph's direct text if present
    if p.text:
        processed_text.append(p.text.strip())

    # Iterate through paragraph and its children to find <w> tags specifically
    for elem in p.iter():
        if elem.tag == 'w':
            # Process <w> tag and append its structured content
            processed_text.append(process_w_tag(elem))
        # Append the tail text of elements (text following a tag within the same element)
        if elem.tail:
            processed_text.append(elem.tail.strip())

    # Join the parts together and clean the text
    text_content = " ".join(processed_text)
    text_content = re.sub('\s+', ' ', text_content).strip()  # Normalize whitespace
    text_content = re.sub(r'\(\?\)', '', text_content)  # Remove question marks in parentheses
    # Use clean_text function to remove quotation marks
    text_content = clean_text(text_content)

    for note in p.findall('.//note'):
        note_text = note.text.strip() if note.text else ""
        target_id = note.get("target", "")
        # Clean note text as well
        note_text = clean_text(note_text)
        text_content = text_content.replace(f'<addSpan to="{target_id}"/>', "")
        text_content = text_content.replace(f'<anchor id="{target_id}"/>', f' ({note_text})')
    return {
        "id": paragraph_id,
        "corresp": corresp,
        "text": text_content
    }


def process_xml_file(xml_file):
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(str(xml_file), parser)
    paragraphs = [process_p_tag(p) for p in tree.findall('.//p')]
    return paragraphs


with output_cleaned_data.open(mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["File", "p_id", "corresp", "text"])

    for xml_file in xml_directory.glob('*.xml'):
        paragraphs_info = process_xml_file(xml_file)
        for info in paragraphs_info:
            if "N/A" in [info["id"], info["corresp"], info["text"]] or not info["text"].strip():
                continue  # Skip this entry
            writer.writerow([xml_file.name, info["id"], info["corresp"], info["text"]])


def process_and_clean_text(text):
    # Remove '+' signs directly associated with numbers
    text = re.sub(r'\s*\+\s*', ' ', text)

    # Replace 'X' used in numeric contexts, treating it as a placeholder for uncertainty
    # Here, 'X' is removed to simplify the process of finding the largest number
    text = re.sub(r'\bX\b', '', text)

    # Find sequences of numbers (with possible spaces after removing 'X') and replace them with the largest number
    def replace_with_largest(match):
        numbers = [int(num) for num in match.group().split()]
        return str(max(numbers)) if numbers else ''

    text = re.sub(r'\b(\d+\s+)+\d+\b', replace_with_largest, text)

    # Fix [DN: Sn] -iddinam into [DN: Sn-iddinam]
    text = re.sub(r'\[DN: Sn\] -iddinam', '[DN: Sn-iddinam]', text)
    # Remove entries like [Label: ] with an empty label for all specified types
    text = re.sub(r'\[(DN|EN|GN|MN|ON|PN|RN|SN|TN|WN): \]', '', text)

    # Normalize whitespace around punctuation
    text = re.sub(r'\s*,\s*', ', ', text)  # Normalize spaces around commas
    text = re.sub(r'\s*!\s*', '! ', text)  # Normalize spaces around exclamation marks
    text = re.sub(r'\s*\.\s*', '. ', text)  # Normalize spaces around periods

    # Correct punctuation issues, including ".," to "."
    text = re.sub(r'\.,', '.', text)
    text = re.sub(r',\s*(,+\s*)+', ', ', text)  # Handle multiple consecutive commas

    # Normalize spacing around colons and correct other punctuation
    text = re.sub(r'(?<!\s):\s*', ': ', text)  # Ensure space after colons
    text = re.sub(r'\s+(:)', r'\1', text)  # Remove space before colons
    text = re.sub(r':\s*,', ': ', text)  # Correct space after colon followed by comma

    # Handle comma followed by a full stop as a single full stop
    text = re.sub(r',\.', '.', text)

    # Remove lone periods and correct sequences
    text = re.sub(r'\.\s+(?=[A-Z][a-z])', ' ', text)  # Lone periods before sentence continuation
    text = re.sub(r'\.\s*,', '.', text)  # Remove comma after a full stop

    # Remove unnecessary periods before commas and handle repetitive periods
    text = re.sub(r'\.\s*,', ',', text)
    text = re.sub(r'(\. )+\.', '.', text)  # Reduce multiple periods to one
    text = re.sub(r'\.(\s*\. )+', '. ', text)  # Correct sequences of periods

    # Ensure no space before punctuation at the end
    text = re.sub(r'\s+([,.!])$', r'\1', text)

    # Remove unnecessary whitespaces
    text = ' '.join(text.split())

    return text


entity_pattern = re.compile(r'\[([A-Z]{2}): ([^\]]+?)(\'s)?\]')
def normalize_entity_forms(text):
    """
    Normalize entity forms by directly modifying annotations to remove possessive 's'.
    """

    def replacement(match):
        entity_tag, entity_name, possessive = match.groups()
        # Normalize entity name by removing possessive 's'
        if possessive:
            return f"[{entity_tag}: {entity_name}]"
        else:
            return match.group()

    normalized_text = re.sub(entity_pattern, replacement, text)

    return normalized_text


def post_process_csv(input_csv, output_cleaned_data):
    with input_csv.open(mode='r', newline='', encoding='utf-8') as infile, output_cleaned_data.open(mode='w', newline='',encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        headers = next(reader)
        writer.writerow(headers)

        for row in reader:
            row[3] = process_and_clean_text(row[3])
            row[3] = normalize_entity_forms(row[3])

            # Skip writing the row if the text content is empty,
            # contains only whitespace, or is just a "."
            if not row[3].strip() or row[3] == ".":
                continue

            writer.writerow(row)


# Unique entities storage
unique_entities = defaultdict(set)

# First, normalize and clean the CSV content
post_process_csv(output_cleaned_data, temp_csv)

# Then, read the processed CSV to extract unique entities
with temp_csv.open(mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        text = row['text']
        for match in entity_pattern.finditer(text):
            entity_tag, entity_name = match.groups()[:2]
            unique_entities[entity_tag].add(entity_name)

# Print the unique entities for each tag
for tag, entities in unique_entities.items():
    print(f"Tag: {tag}")
    print("Entities:", ", ".join(sorted(entities)))
    print("---\n")

output_cleaned_data.unlink()  # Remove the original file
temp_csv.rename(output_cleaned_data)
print(f"Data has been written to {output_cleaned_data}")
