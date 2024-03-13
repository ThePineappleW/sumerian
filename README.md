# English - Sumerian Translation

## :bulb: Project Overview
This project aims to leverage Natural Language Processing (NLP) techniques to enhance the translation of ancient and low-resource languages. Specifically, it targets two significant challenges: the creation of a dynamic, bidirectional translator for transliterations between Sumerian and English, and the development of a Named Entity Recognition (NER) system to address lexical gaps in English caused by the translation of Sumerian language specific proper nouns.

### Sumerian Entity Recognizer
![image](https://github.com/ThePineappleW/sumerian/assets/53830950/9a654810-91f1-4dea-9a6d-ee1f63c02f5e)
![image](https://github.com/ThePineappleW/sumerian/assets/53830950/9ef021e3-9981-4a1a-b3d9-87666a231f5d)
### English - Sumerian
https://github.com/ThePineappleW/sumerian/assets/53830950/2557fd69-dc89-4ee2-8a96-fbcccc76aba4
### Sumerian - English
https://github.com/ThePineappleW/sumerian/assets/53830950/f799dc0e-8b5e-4b39-bae1-f5f6bd55ef4f

## :gear: Methodology
### Objective:
- To build functional classification-translation models for translating Sumerian transliterations to English and vice versa, as well as to identify and classify Sumerian entities within English translations, addressing the challenge of lexical gaps due to the historical and cultural distance of the language.

### Techniques and Process:
- Bidirectional LSTM Networks: Utilized LSTM networks to handle the language translation sequence-to-sequence challenges and maintain context between input terms.
- Subword Tokenization: Implemented different tokenization methods (BPE, Unigram, custom delimiter-based) using SentencePiece to manage the agglutinative nature of Sumerian.
- POS Tagging and Agglomerative Clustering: Integrated linguistic features such as part of speech tags and semantically grouped word clusters to improve translation accuracy.
- spaCy CNN: Developed a NER model using spaCy's NLP tools to classify and recognize ten categories of Sumerian ideophonic entities such as deities, ethnonyms, and geographic names which do not have direct modern equivalents.
- Contextual Accuracy Enhancement: Implemented the utilization of contextual information to improve accuracy in token classification and translation.

## :books: Data Resources
- The data come from the ETCSL corpus. [Manual](https://etcsl.orinst.ox.ac.uk/edition2/etcslmanual.php)

## :books: Sumerian Resources
- [Grammar of Sumerian](https://scholarlypublications.universiteitleiden.nl/handle/1887/16107) (Warning: *very* hefty and academic)
- [An Introduction to the Grammar of Sumerian](https://edit.elte.hu/xmlui/bitstream/handle/10831/31083/ZolyomiG_Introduction%20to%20the%20grammar%20of%20Sumerian.pdf;sequence=1) (Much more approachable)

---

<sub>Project Contributors: Ethan Rogers, Indrajeet Aditya Roy, Emery Jacobowitz</sub>
