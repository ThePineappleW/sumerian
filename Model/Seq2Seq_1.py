import json
import numpy as np
import tensorflow as tf
import csv
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the combined corpus of Sumerian transliterations and their English translations
with open('combined_corpus.json', 'r') as file:
    data = json.load(file)

# Define a function to process lexical information for each entry
def process_lexical_info(entry):
    # Extract transliteration and detailed word information
    transliteration_words = entry['transliteration_words']
    transliteration = []
    for word_info in transliteration_words:
        # Extract the original word, its part of speech (POS) tag, and lemma, defaulting to 'UNK' if not available
        word = word_info['original']
        pos = word_info['attributes'].get('pos', 'UNK')
        lemma = word_info['attributes'].get('lemma', 'UNK')
        # Combine the original word with its POS and lemma for extra detailed transliteration
        enriched_word = f"{word}({pos}:{lemma})"
        transliteration.append(enriched_word)
    return ' '.join(transliteration)

# Process the corpus to for extra transliteration detail
sumerian_transliterations = [process_lexical_info(entry) for entry in data]
# Process translations to include start and end tokens
english_translations = ['<start> ' + entry['translation'] + ' <end>' for entry in data]


# Load the glossary as a dictionary to assist with translating individual Sumerian words
glossary = {}
with open('translation_glossary_dataset.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        glossary[row['original']] = row['translation_label']

# Define a function to augment Sumerian transliterations with their English equivalents using the glossary
def augment_with_glossary(sumerian_text, glossary):
    words = sumerian_text.split()
    augmented_text = [glossary.get(word, word) for word in words]
    return ' '.join(augmented_text)

# Augment transliterations with English equivalents
augmented_sumerian_transliterations = [augment_with_glossary(text, glossary) for text in sumerian_transliterations]

# Tokenization: Convert texts to sequences of integers
sumerian_tokenizer = Tokenizer(filters='')
sumerian_tokenizer.fit_on_texts(augmented_sumerian_transliterations)
sumerian_sequences = sumerian_tokenizer.texts_to_sequences(augmented_sumerian_transliterations)

english_tokenizer = Tokenizer(filters='')
english_tokenizer.fit_on_texts(english_translations)
english_sequences = english_tokenizer.texts_to_sequences(english_translations)

# Pad sequences to ensure uniform length
sumerian_padded = pad_sequences(sumerian_sequences, padding='post')
english_padded = pad_sequences(english_sequences, padding='post')

# Model parameters
vocab_size_src = len(sumerian_tokenizer.word_index) + 1  # Source vocabulary size
vocab_size_tgt = len(english_tokenizer.word_index) + 1  # Target vocabulary size
lstm_units = 256  # Number of units in LSTM layer

# Encoder

# Input layer: Accepts sequences of arbitrary length (None indicates variable length) with integer encodings of Sumerian words.
encoder_inputs = Input(shape=(None,))

# Embedding layer: Maps each integer in the sequence to a dense vector of fixed size ('lstm_units'). 
# This representation captures semantic information about the words.
encoder_embedding = Embedding(vocab_size_src, lstm_units)(encoder_inputs)

# LSTM layer: Processes the sequence of word embeddings to extract the context. 
# 'return_sequences=True' ensures the output includes all hidden states, which are needed for attention.
# 'return_state=True' returns the final hidden and cell states, capturing the context of the entire sequence.
encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)

# Execution of the LSTM layer. 
# 'encoder_outputs' contains the sequence of hidden states (used for attention).
# 'state_h' and 'state_c' are the final hidden and cell states, summarizing the input sequence.
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# The final states serve as the initial context for the decoder.
encoder_states = [state_h, state_c]


# Decoder

# Input layer: Similar to the encoder, it accepts sequences of arbitrary length, representing the target language (English).
decoder_inputs = Input(shape=(None,))

# Embedding layer: Maps each integer (English word encoding) to a dense vector. 
# This is analogous to the encoder embedding but uses the target language vocabulary.
decoder_embedding = Embedding(vocab_size_tgt, lstm_units)(decoder_inputs)

# LSTM layer: The decoder LSTM is initialized with the encoder's final states, linking the encoding and decoding processes.
# It also returns sequences and states because we need the sequence of outputs for attention and potentially deeper layers.
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)

# Execution of the LSTM layer, using the encoder's final states as the initial states.
# The '_' placeholders are for the final hidden and cell states, which are not used directly here but might be in more complex models.
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)


# Instantiate the custom Attention layer with a specified number of units.
# This layer will be used to compute context vectors and attention weights.
attention_layer = Attention(lstm_units)

# Compute the context vector and attention weights
# The attention layer takes in the final hidden state of the encoder ('state_h') as the 'query'
# and the full sequence of encoder outputs as 'values'.
# 'state_h' serves as a summary of the input sequence up to the current decoding step,
# guiding the attention mechanism to focus on specific parts of the input sequence.
context_vector, attention_weights = attention_layer(state_h, encoder_outputs)

# Expanding the context vector to match the decoder output's dimensions
context_vector_expanded = tf.expand_dims(context_vector, 1)
# Tiling the expanded context vector across the sequence length of the decoder outputs
# This ensures that each step of the decoder output has a corresponding context vector.
context_vector_tiled = tf.tile(context_vector_expanded, [1, tf.shape(decoder_outputs)[1], 1])
# Concatenate the tiled context vector with the decoder outputs along the last dimension
# This combines the context-specific information with the decoder's predictions at each step in the sequence.
decoder_combined_context = Concatenate(axis=-1)([decoder_outputs, context_vector_tiled])


# Define the output layer
# This Dense layer converts the decoder output into a probability distribution over the target vocabulary.
decoder_dense = Dense(vocab_size_tgt, activation='softmax')
# Apply the Dense layer to the combined context and decoder output
# Each position in the output sequence is now associated with a probability distribution over the target vocabulary.
decoder_outputs = decoder_dense(decoder_combined_context)


# Compile the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize a zero matrix with the same shape as the padded English sequences.
decoder_input_data = np.zeros_like(english_padded)
# Shift the sequences to the right by one time step.
# This operation fills the sequence from the second position onwards with the original data,
# effectively adding a 'start of sequence' token at the beginning of each sequence.
decoder_input_data[:, 1:] = english_padded[:, :-1]
# Insert the '<start>' token's index at the beginning of each sequence.
# This token is crucial for signaling the decoder to start generating the translation.
decoder_input_data[:, 0] = english_tokenizer.word_index['<start>']
# Expand the dimensions of the target data to fit the expected model output.
# This operation adds an extra dimension to the target data, making it compatible with
# the sparse categorical cross-entropy loss function used during model training.
decoder_target_data = np.expand_dims(english_padded, -1)

# Train the model
model.fit([sumerian_padded, decoder_input_data], decoder_target_data, batch_size=64, epochs=100, validation_split=0.2)

# Save the trained model and tokenizers for future use
model.save('seq2seq_sumerian_english.h5')
with open('sumerian_tokenizer.pkl', 'wb') as file:
    pickle.dump(sumerian_tokenizer, file)
with open('english_tokenizer.pkl', 'wb') as file:
    pickle.dump(english_tokenizer, file)