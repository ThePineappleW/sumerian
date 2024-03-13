import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import concatenate, Dense, Dropout, Embedding, Input, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('../processed_transliteration_data.csv')
# Perform one-hot encoding on the 'Category' column
categories_one_hot = pd.get_dummies(df['category'], prefix='category')
category_one_hot_array = categories_one_hot.to_numpy()

# Initialize tokenizers
english_tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
sumerian_tokenizer = Tokenizer(filters='!"#$%&()*+.,/:;=?@[\\]^`{|}~\t\n ')
pos_tokenizer = Tokenizer()

# Fit tokenizers on the respective columns
sumerian_tokenizer.fit_on_texts(df['lemma'])
pos_tokenizer.fit_on_texts(df['pos'])
english_tokenizer.fit_on_texts(df['label'])

# Tokenize and pad sequences for the Sumerian lemmas and POS tags
sumerian_sequences = sumerian_tokenizer.texts_to_sequences(df['lemma'])
pos_sequences = pos_tokenizer.texts_to_sequences(df['pos'])
english_sequences = english_tokenizer.texts_to_sequences(df['label'])
# Determine max sequence length from the inputs
max_seq_length = max([len(seq) for seq in sumerian_sequences])

# Pad the sequences to the maximum sequence length
sumerian_padded = pad_sequences(sumerian_sequences, maxlen=max_seq_length, padding='post')
pos_padded = pad_sequences(pos_sequences, maxlen=max_seq_length, padding='post')
english_padded = pad_sequences(english_sequences, maxlen=max_seq_length, padding='post')

# Define vocabulary sizes
sumerian_vocab_size = len(sumerian_tokenizer.word_index) + 1
pos_vocab_size = len(pos_tokenizer.word_index) + 1
category_vocab_size = categories_one_hot.shape[1]
english_vocab_size = len(english_tokenizer.word_index) + 1
# Model architecture
input_lemma = Input(shape=(max_seq_length,), dtype='int32', name='input_lemma')
input_pos = Input(shape=(max_seq_length,), dtype='int32', name='input_pos')
input_category = Input(shape=(category_vocab_size,), dtype='float32', name='input_category')

sumerian_embedding = Embedding(input_dim=sumerian_vocab_size, output_dim=256, name='sumerian_embedding')(input_lemma)
pos_embedding = Embedding(input_dim=pos_vocab_size, output_dim=256, name='pos_embedding')(input_pos)
combined_embeddings = concatenate([sumerian_embedding, pos_embedding], axis=-1, name='combined_embeddings')

# Prepare category one-hot encoding
category_one_hot_encoded = np.repeat(categories_one_hot.values[:, np.newaxis, :], max_seq_length, axis=1)
category_repeated = RepeatVector(max_seq_length)(input_category)
combined_with_category = concatenate([combined_embeddings, category_repeated], axis=-1, name='combined_with_category')

lstm_layer = LSTM(512, return_sequences=False, name='lstm_layer')(combined_with_category)
lstm_layer = Dropout(0.5)(lstm_layer)
output_layer = Dense(units=english_vocab_size, activation='softmax', name='output_layer')(lstm_layer)

model = Model(inputs=[input_lemma, input_pos, input_category], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Prepare the target labels for training
y = np.array([seq[0] for seq in english_padded])
# Split the padded Sumerian and POS sequences and labels into training and test sets
X_train_sumerian, X_test_sumerian, X_train_pos, X_test_pos, y_train, y_test = train_test_split(sumerian_padded,
                                                                                               pos_padded, y,
                                                                                               test_size=0.2,
                                                                                               random_state=42)
# Split the one-hot encoded category data to match the training and testing sets
X_train_category, X_test_category = train_test_split(category_one_hot_array, test_size=0.2, random_state=42)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    [X_train_sumerian, X_train_pos, X_train_category],
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=([X_test_sumerian, X_test_pos, X_test_category], y_test),
    callbacks=[early_stopping]
)

history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs = range(1, len(train_loss) + 1)
# Plotting training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model.save('sumerian_translation_model.keras')

y_pred = model.predict([X_test_sumerian, X_test_pos, X_test_category])
y_pred_classes = np.argmax(y_pred, axis=1)
test_f1_score = f1_score(y_test, y_pred_classes, average='micro')
test_accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test F1 Score: {test_f1_score}")
print(f"Test Accuracy: {test_accuracy}")

with open('sumerian_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(sumerian_tokenizer.to_json())

with open('english_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(english_tokenizer.to_json())

with open('pos_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(pos_tokenizer.to_json())
