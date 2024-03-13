import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Concatenate, Dense, Embedding, Input, LSTM, RepeatVector, Reshape, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('../categorized_consolidated_transliteration_data.csv')
# Initialize tokenizers
english_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`_{|}~\t\n')
sumerian_tokenizer = Tokenizer(filters='!"#$%&()*+.,/:;=?@[\\]^`{|}~\t\n ')
pos_tokenizer = Tokenizer()

# Fit tokenizers on the new texts
english_tokenizer.fit_on_texts(df['label'])
sumerian_tokenizer.fit_on_texts(df['lemma'])
pos_tokenizer.fit_on_texts(df['pos'])

# Convert texts to sequences
english_seq = english_tokenizer.texts_to_sequences(df['label'])
sumerian_seq = sumerian_tokenizer.texts_to_sequences(df['lemma'])
pos_seq = pos_tokenizer.texts_to_sequences(df['pos'])

# Pad the sequences
max_seq_length = max(max(len(seq) for seq in english_seq), max(len(seq) for seq in sumerian_seq))
english_padded = pad_sequences(english_seq, maxlen=max_seq_length, padding='post')
sumerian_padded = pad_sequences(sumerian_seq, maxlen=max_seq_length, padding='post')
pos_padded = pad_sequences(pos_seq, maxlen=max_seq_length, padding='post')
# Prepare the language input feature
language_seq = np.concatenate([df['Language_English'].values.reshape(-1, 1), df['Language_Sumerian'].values.reshape(-1, 1)], axis=1)

# Vocabulary sizes
english_vocab_size = len(english_tokenizer.word_index) + 1
sumerian_vocab_size = len(sumerian_tokenizer.word_index) + 1
pos_vocab_size = len(pos_tokenizer.word_index) + 1
category_vocab_size = df['Category'].nunique() + 1

# Define input layers
english_input = Input(shape=(max_seq_length,), dtype='int32', name='english_input')
pos_input = Input(shape=(max_seq_length,), dtype='int32', name='pos_input')
category_input = Input(shape=(1,), dtype='int32', name='category_input')
language_input = Input(shape=(2,), dtype='float32', name='language_input')

# Embedding layers
english_embedding = Embedding(input_dim=english_vocab_size, output_dim=100)(english_input)
pos_embedding = Embedding(input_dim=pos_vocab_size, output_dim=25)(pos_input)
category_embedding_layer = Embedding(input_dim=category_vocab_size, output_dim=10)(category_input)
category_embedding = Reshape((10,))(category_embedding_layer)

# Repeat the category and language embeddings to match the sequence length
category_embedding_repeated = RepeatVector(max_seq_length)(category_embedding_layer)
language_embedding_repeated = RepeatVector(max_seq_length)(language_input)
combined_embeddings = concatenate([english_embedding, pos_embedding, category_embedding_repeated, language_embedding_repeated], axis=-1)

# LSTM layer
lstm_layer = LSTM(units=256, return_sequences=True)(combined_embeddings)
# Dense output layer for Sumerian prediction
output_layer = TimeDistributed(Dense(units=sumerian_vocab_size, activation='softmax'))(lstm_layer)
# Build and compile the model
model = Model(inputs=[english_input, pos_input, category_input, language_input], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Prepare labels (Sumerian sequences) for training
y = sumerian_padded.reshape(*sumerian_padded.shape, 1)
# Split into training and testing sets
X_english_train, X_english_test, X_pos_train, X_pos_test, X_category_train, X_category_test, X_language_train, X_language_test, y_train, y_test = train_test_split(
    english_padded, pos_padded, df['Category'].values, language_seq, y, test_size=0.2, random_state=42)
# Fit the model
model.fit(
    {'english_input': X_english_train, 'pos_input': X_pos_train, 'category_input': X_category_train,
     'language_input': X_language_train},
    y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.1
)
model.save('english_translation_model.keras')

test_loss, test_accuracy = model.evaluate(
    {'english_input': X_english_test, 'pos_input': X_pos_test, 'category_input': X_category_test,
     'language_input': X_language_test},
    y_test
)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
