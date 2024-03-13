#! /usr/bin/python3
from bs4 import BeautifulSoup
import argparse
from pathlib import Path
import glob
import re
import json

def clean_word(word: str):
    """
    The transliteration standards include a number of special characters.
    This function "handles" them.
    https://etcsl.orinst.ox.ac.uk/edition2/etcslmanual.php#char
    """
    # TODO
    word = word.strip()
    # Remove punctuation
    word = re.sub(r'&(X|hr);', '', word)
    # Remove damage, supplied, qry, and subscript
    word = re.sub(r'&(dam|supp|qry|sub)b;.*?&\1e;', '', word)
    # Remove 'x': this is used for damaged words, and is not in SUX otherwise.
    word = re.sub(r'x', '', word)
    # Fix spacings
    word = re.sub(r'\s+', ' ', word)
    word = word.lower()
    return word


def transliteration_to_str(filename):
    """
    Reads the given transliteration xml file and prints its contents in plaintext.
    Discards grammatical information.
    """

    with open(filename, 'r') as xml:
        doc = BeautifulSoup(xml, 'lxml')
    
    lines = doc.find_all('l')
    for line in lines:
        print(line_to_str(line))

def line_to_str(line):
    words = [clean_word(word.text) for word in line('w') if word != '\n']
    return ' '.join(words).strip()
    

def translation_to_json(translation_filename, transliteration_dir, print_json=True):
    """
    Given a file containing ETCSL translations, 
    aggregates translations and transliterations in a TSV file
    where a translation is followed by all of the relevant transliterated words.

    Note that in the ETCSL, transliterations are separated by line, while translations are separated by paragraph.
    The output of this script will ignore transliteration lines.
    """

    translat = Path(translation_filename)

    with open(translation_filename, 'r') as translat_xml:
        with open(Path(transliteration_dir) / f'c{translat.name[1:]}', 'r') as translit_xml:
            translat_doc = BeautifulSoup(translat_xml, 'lxml')
            translit_doc = BeautifulSoup(translit_xml, 'lxml')

    paragraphs = translat_doc.find_all('p', attrs={'corresp' : True})
    for paragraph in paragraphs:
        if paragraph.text != '' and paragraph.get('id', 'X') != 'X':
            lines = translit_doc.find_all(attrs={'corresp' : paragraph['id']})
            output = '\n'.join([line_to_str(line) for line in lines])
            if print_json:
                print(json.dumps({'transliteration' : output, 'translation': paragraph.text}))
            else:
                print(f'{paragraph.text}\n\n{output}', end=f'\n\n{"="*50}\n\n')





def main(args):
    if args.recursive:
        for f in Path(args.infile).glob('**/*.xml'):
            translation_to_json(f, args.translit, print_json=args.json)
        
    else: 
        translation_to_json(args.infile, args.translit, print_json=args.json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='etcsl_utils',
        description='Various functions for ETCSL data')
    
    parser.add_argument('-i', '--infile', type=str, required=True, help='The file or directory from which to read')
    parser.add_argument('-t', '--translit', type=str, default='./ETCSL/transliterations/', help='The transliteration directory')
    parser.add_argument('-j', '--json', action=argparse.BooleanOptionalAction, default=False, help='Print the output as JSONlines')
    parser.add_argument('-r', '--recursive', action=argparse.BooleanOptionalAction, default=False, help='Read all XML files in a directory (recursive)')
    parser.add_argument('-e', '--exp', action=argparse.BooleanOptionalAction, default=False, help='Experimental features')

    args = parser.parse_args()
    main(args)