import numpy as np
import pandas as pd
import sentencepiece as spm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from keras.layers import Concatenate, Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.src.callbacks import EarlyStopping
from keras_nlp.tokenizers import SentencePieceTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('../categorized_consolidated_transliteration_data.csv')
# The SUX tokenizer uses a pretrained model which made from the CLI in `Data/sentencePiece/`.
# This can also be accomplished using the python interface:
# https://github.com/google/sentencepiece/blob/master/python/README.md
SPM_FILE = './Data/sentencePiece/suxBPE'
# Initialize tokenizers
english_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
# sumerian_tokenizer = Tokenizer(filters='!"#$%&()*+.,/:;=?@[\\]^`{|}~\t\n ')
sumerian_tokenizer = SentencePieceTokenizer(proto=SPM_FILE)
pos_tokenizer = Tokenizer()

# Fit tokenizers on the respective columns
# sumerian_tokenizer.fit_on_texts(df['form'])
pos_tokenizer.fit_on_texts(df['pos'])
english_tokenizer.fit_on_texts(df['label'])

# Tokenize and pad sequences for the Sumerian lemmas, POS tags, and categories
sumerian_sequences = sumerian_tokenizer.tokenize(df['form']).to_tensor()
# sumerian_sequences = sumerian_tokenizer.texts_to_sequences(df['form'])
pos_sequences = pos_tokenizer.texts_to_sequences(df['pos'])
category_sequences = df['Category'].values
# Determine max sequence length from the inputs
max_seq_length = max([len(seq) for seq in sumerian_sequences])

# Pad the sequences to the maximum sequence length
sumerian_padded = pad_sequences(sumerian_sequences, maxlen=max_seq_length, padding='post')
pos_padded = pad_sequences(pos_sequences, maxlen=max_seq_length, padding='post')
category_padded = pad_sequences(category_sequences.reshape(-1, 1), maxlen=max_seq_length, padding='post')
# Convert English labels to sequences and pad
english_sequences = english_tokenizer.texts_to_sequences(df['label'])
english_padded = pad_sequences(english_sequences, maxlen=max_seq_length, padding='post')

# Define vocabulary sizes
sumerian_vocab_size = sumerian_tokenizer.vocabulary_size() + 1
# sumerian_vocab_size = len(sumerian_tokenizer.word_index) + 1
pos_vocab_size = len(pos_tokenizer.word_index) + 1
category_vocab_size = df['Category'].nunique() + 1
english_vocab_size = len(english_tokenizer.word_index) + 1

# Define the model
input_lemma = Input(shape=(max_seq_length,), dtype='int32', name='input_lemma')
input_pos = Input(shape=(max_seq_length,), dtype='int32', name='input_pos')
input_category = Input(shape=(max_seq_length,), dtype='int32', name='input_category')

sumerian_embedding = Embedding(sumerian_vocab_size, 100)(input_lemma)
pos_embedding = Embedding(pos_vocab_size, 25)(input_pos)
category_embedding = Embedding(category_vocab_size, 10)(input_category)
combined_embeddings = concatenate([sumerian_embedding, pos_embedding, category_embedding], axis=-1)

lstm_layer = LSTM(256, return_sequences=False)(combined_embeddings)
output_layer = Dense(english_vocab_size, activation='softmax')(lstm_layer)
model = Model(inputs=[input_lemma, input_pos, input_category], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

y = np.array([seq[0] for seq in english_sequences])
# Split the data into training and testing sets
X_train_sumerian, X_test_sumerian, X_train_pos, X_test_pos, X_train_category, X_test_category, y_train, y_test = train_test_split(
    sumerian_padded, pos_padded, category_padded, y, test_size=0.2, random_state=42)

# Add an EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Train the model
history = model.fit(
    [X_train_sumerian, X_train_pos, X_train_category],
    y_train,
    batch_size=64,
    epochs=50,
    validation_data=([X_test_sumerian, X_test_pos, X_test_category], y_test),
    callbacks=[early_stopping],
    verbose=2
)

model.save('sumerian_translation_model.keras')
y_pred = model.predict([X_test_sumerian, X_test_pos, X_test_category])
y_pred_classes = np.argmax(y_pred, axis=1)

test_f1_score = f1_score(y_test, y_pred_classes, average='micro')
test_accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test F1 Score: {test_f1_score}")
print(f"Test Accuracy: {test_accuracy}")

with open('english_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(english_tokenizer.to_json())

with open('pos_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(pos_tokenizer.to_json())
