import gradio as gr
import spacy
from spacy import displacy

colors = {
    "DEITY": "#33FFE9",
    "ETHNONYM": "#ff006e",
    "GEOGRAPHIC": "#0096c7",
    "MONTH": "#ffd60a",
    "OBJECT": "#ef476f",
    "PERSON": "#FF3333",
    "ROYAL": "#f77f00",
    "SETTLEMENT": "#6EFF33",
    "TOPONYM": "#fb5607",
    "WATERCOURSE": "#FFF033",
}

options = {
    "ents": ["DEITY", "ETHNONYM", "GEOGRAPHIC", "MONTH", "OBJECT", "PERSON", "ROYAL", "SETTLEMENT", "TOPONYM", "WATERCOURSE"],
    "colors": colors
}

model_dir = "ner_1"
nlp = spacy.load(model_dir)

def visualize_ner(text):
    doc = nlp(text)
    html = displacy.render(doc, style="ent", options=options, minify=True, page=True)
    return html

iface = gr.Interface(fn=visualize_ner, inputs="text", outputs="html", title="Sumerian Entity Recognition",allow_flagging="never")
iface.launch()
