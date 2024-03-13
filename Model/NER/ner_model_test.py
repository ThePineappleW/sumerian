import spacy
from spacy.training import Example

model_dir = "ner_1"
nlp = spacy.load(model_dir)

TEST_DATA = [
    (
    "The king, ulgi, the good shepherd of Sumer, his feet upon ; he took his seat on a throne of The and drums resounded for him, and the drums played music for him.",
    {'entities': [(10, 14, 'ROYAL'), (37, 42, 'GEOGRAPHIC')]}),
    (
    ", sang the singers for him in a song His boatmen, in tireless effort, These, citizens of Enegir and citizens of Urim, thrust forth their oars at the command of the lord He moored the boat at the temple area of Nibru, the temple area Dur-an-ki, at Enlil Kar-etina He entered before Enlil with the silver and lapis lazuli of the foreign lands loaded into leather pouches and leather bags, all their heaped-up treasures, and with the amassed wealth of the foreign lands.",
    {'entities': [(89, 95, 'SETTLEMENT'), (112, 116, 'SETTLEMENT'), (210, 215, 'SETTLEMENT'), (233, 242, 'TOPONYM'),
                  (247, 252, 'DEITY'), (253, 262, 'SETTLEMENT'), (281, 286, 'DEITY')]}),
    (
    "How come you did not know how long it would take to make Ibi- return to the mountain lands? Why have you and Erra Girbubu, the governor of irikal, not confronted him with the troops which you had at hand? How could you allow him to restore ?",
    {'entities': [(57, 61, 'ROYAL'), (109, 113, 'DEITY'), (114, 121, 'PERSON'), (139, 145, 'SETTLEMENT')]}),
    (
    "By hand Winter guided the spring floods, the abundance and life of the Land, down from the edge of the hills He set his foot upon the Tigris and Euphrates like a big bull and released them into the fields and fruitful acres of Enlil He shaped lagoons in the sea He let fish and birds together come into existence by the sea He surrounded all the reedbeds with mature reeds, reed shoots and reeds.",
    {'entities': [(134, 140, 'WATERCOURSE'), (145, 154, 'WATERCOURSE'), (227, 232, 'DEITY')]}),
    ("When I was setting out, their from the bank of the Ab-gal watercourse to the province of Zimudar.",
     {'entities': [(51, 57, 'WATERCOURSE'), (89, 96, 'GEOGRAPHIC')]}),
    (
    "You are the light of the good shepherd Enlil, and you have been given a majestic name by Ninlil You have been given wisdom by Enki You were born to Enul and Ninul, and so you are united with the lordly seed You are the E-kur song You are a minister fit for his king: Nuska, you are the man of Enlil heart.",
    {'entities': [(39, 44, 'DEITY'), (89, 95, 'DEITY'), (126, 130, 'DEITY'), (148, 152, 'DEITY'), (157, 162, 'DEITY'),
                  (219, 224, 'TOPONYM'), (267, 272, 'DEITY'), (293, 298, 'DEITY')]}),
    (
    "An frightened the very dwellings of Sumer, the people were afraid. Enlil blew an evil storm, silence lay upon the city. Nintur bolted the door of the storehouses of the Land. Enki blocked the water in the Tigris and the Euphrates. Utu took away the pronouncement of equity and justice. Inana handed over victory in strife and battle to a rebellious land. Ninirsu poured Sumer away like milk to the dogs Turmoil descended upon the Land, something that no one had ever known, something unseen",
    {'entities': [(0, 2, 'DEITY'), (36, 41, 'GEOGRAPHIC'), (67, 72, 'DEITY'), (120, 126, 'DEITY'), (175, 179, 'DEITY'),
                  (205, 211, 'WATERCOURSE'), (220, 229, 'WATERCOURSE'), (231, 234, 'DEITY'), (286, 291, 'DEITY'),
                  (355, 362, 'DEITY'), (370, 375, 'GEOGRAPHIC')]}),
    (
        "ulgi, the shepherd is the honey man beloved by Nibru ; may the true shepherd, ulgi, refresh himself in the pleasant shade of Enlil brickwork!",
        {'entities': [(0, 4, 'ROYAL'), (47, 52, 'SETTLEMENT'), (78, 82, 'DEITY'), (125, 130, 'DEITY')]}),
    ("O ulgi, Enlil has brought forth happy days for you in your reign!",
     {'entities': [(2, 6, 'ROYAL'), (8, 13, 'DEITY')]}),
    (
        "Heaven's king, earth's great mountain, Father Enlil, heaven's king, earth's great mountain, thought up something great: he chose ulgi in his heart for a good reign!",
        {'entities': [(46, 51, 'DEITY'), (129, 133, 'DEITY')]}),
    ("An of Enlil.", {'entities': [(6, 11, 'DEITY')]}),
    (
        "There were three friends, citizens of Adab, who fell into a dispute with each other, and sought justice They deliberated the matter with many words, and went before the king.",
        {'entities': [(38, 42, 'SETTLEMENT')]}),
    (
        "When the king came out from the cloistered lady's presence, each man's heart was dissatisfied The man who hated his wife left his wife The man his abandoned his With elaborate words, with elaborate words, the case of the citizens of Adab was settled. Pa-niin-ara, their sage, the scholar, the god of Adab, was the scribe.",
        {'entities': [(233, 237, 'SETTLEMENT'), (251, 262, 'DEITY'), (300, 304, 'SETTLEMENT')]}),
    (
        "You who bundle together the divine powers, the divine powers, articulate house of the king, who give instruction throughout the breadth of heaven and earth, adviser of the Land, Nuska! The Great Mountain Enlil has summoned you to his divine powers He has made long life issue gloriously in heaven and earth for you who were fathered by Lord Nunamnir ; you are his beloved lord He has entrusted the princely divine powers of the E-kur, the august shrine, the holy divine powers, the august and most complex divine powers, the divine powers of the father, of the Great Mountain to you Lord Nuska, summoned by the Prince! He has truly installed you Nuska as leader of the assembly, and has truly installed you to make most brilliant the holy precinct and the pure lustrations, to position the holy vessels, to perfect the divine powers of his status as Enlil, and to amplify the great divine powers.",
        {'entities': [(178, 183, 'DEITY'), (204, 209, 'DEITY'), (341, 349, 'DEITY'), (428, 433, 'TOPONYM'),
                      (588, 593, 'DEITY'), (646, 651, 'DEITY'), (850, 855, 'DEITY')]}),
    (
        "The Great Mountain has entrusted you with organising the divine plans of heaven and earth, throughout the breadth of heaven and earth, setting on their course the great decisions and perfecting the cultic ordinances, Nuska, good lord of Enlil! Impressively strong minister of Enlil, wielding the holy sceptre, pre-eminent leader of the gods, who broadens heaven and earth, good minister, lord of the great words, honourable son of An, with broad chest, endowed with great strength by the Prince, perfecting the divine powers of all that is great! Cup-bearer who makes the holy copper bowls shine, lord of the divine powers of the offering-table, you of great terrifying splendour! Temple cleaner, priest of the, you sprinkle the temple courtyard! Great, working industriously on the Holy Mound to prepare best butter and best milk, reciting to cool the with incantation formulae, to perfect the holy prayers, making shine, hurrying about, organising food offerings,",
        {'entities': [(217, 222, 'DEITY'), (237, 242, 'DEITY'), (276, 281, 'DEITY'), (431, 433, 'DEITY')]})
]

def evaluate_ner(model, test_data):
    scorer = spacy.scorer.Scorer(model)
    examples = []
    for input_text, annotations in test_data:
        doc = model.make_doc(input_text)
        reference = Example.from_dict(doc, annotations)
        prediction = model(input_text)
        example = Example(prediction, reference.y)
        examples.append(example)
    scores = scorer.score(examples)
    return scores

results = evaluate_ner(nlp, TEST_DATA)
print(f"Precision: {results['ents_p']}")
print(f"Recall: {results['ents_r']}")
print(f"F1-score: {results['ents_f']}")

if 'ents_per_type' in results:
    for ent_type, scores in results['ents_per_type'].items():
        print(f"{ent_type} - Precision: {scores['p']}, Recall: {scores['r']}, F1: {scores['f']}")
