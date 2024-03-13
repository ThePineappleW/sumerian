import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from keras.src.layers import RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('../processed_transliteration_data.csv')
# One-hot encoding for categories
categories_one_hot = pd.get_dummies(df['category'], prefix='category')
# Initialize tokenizers
english_tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
sumerian_tokenizer = Tokenizer(filters='!"#$%&()*+.,/:;=?@[\\]^`{|}~\t\n ')
pos_tokenizer = Tokenizer()

# Fit tokenizers on the respective columns
english_tokenizer.fit_on_texts(df['label'])
sumerian_tokenizer.fit_on_texts(df['lemma'])
pos_tokenizer.fit_on_texts(df['pos'])

# Tokenize and pad sequences
english_sequences = english_tokenizer.texts_to_sequences(df['label'])
sumerian_sequences = sumerian_tokenizer.texts_to_sequences(df['lemma'])
pos_sequences = pos_tokenizer.texts_to_sequences(df['pos'])
max_seq_length = max(max(len(seq) for seq in english_sequences), max(len(seq) for seq in sumerian_sequences))
english_padded = pad_sequences(english_sequences, maxlen=max_seq_length, padding='post')
sumerian_padded = pad_sequences(sumerian_sequences, maxlen=max_seq_length, padding='post')
pos_padded = pad_sequences(pos_sequences, maxlen=max_seq_length, padding='post')

# Vocabulary sizes
english_vocab_size = len(english_tokenizer.word_index) + 1
sumerian_vocab_size = len(sumerian_tokenizer.word_index) + 1
pos_vocab_size = len(pos_tokenizer.word_index) + 1
category_vocab_size = categories_one_hot.shape[1]

# Model architecture
input_english = Input(shape=(max_seq_length,), dtype='int32', name='input_english')
english_embedding = Embedding(input_dim=english_vocab_size, output_dim=256, name='english_embedding')(input_english)
input_pos = Input(shape=(max_seq_length,), dtype='int32', name='input_pos')
pos_embedding = Embedding(input_dim=pos_vocab_size, output_dim=256, name='pos_embedding')(input_pos)
combined_embeddings = concatenate([english_embedding, pos_embedding], axis=-1, name='combined_embeddings')

input_category = Input(shape=(category_vocab_size,), dtype='float32', name='input_category')
category_repeated = RepeatVector(max_seq_length)(input_category)
combined_with_category = concatenate([combined_embeddings, category_repeated], axis=-1, name='combined_with_category')

lstm_layer = LSTM(512, return_sequences=True, name='lstm_layer')(combined_with_category)
dropout_layer = Dropout(0.5)(lstm_layer)
output_layer = Dense(units=sumerian_vocab_size, activation='softmax', name='output_layer')(dropout_layer)
# Create and compile the model
model = Model(inputs=[input_english, input_pos, input_category], outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Prepare the target labels for training
y = pad_sequences(sumerian_sequences, maxlen=max_seq_length, padding='post')
X_train_english, X_test_english, X_train_pos, X_test_pos, X_train_category, X_test_category, y_train, y_test = train_test_split(
    english_padded, pos_padded, categories_one_hot.values, y, test_size=0.2, random_state=42)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Train the model
history = model.fit(
    [X_train_english, X_train_pos, X_train_category],
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=([X_test_english, X_test_pos, X_test_category], y_test),
    callbacks=[early_stopping],
    verbose=2
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

model.save('english_translation_model.keras')

y_pred = model.predict([X_test_english, X_test_pos, X_test_category])
y_test_flat = y_test.flatten()
y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
test_f1_score = f1_score(y_test_flat, y_pred_flat, average='micro')
test_accuracy = accuracy_score(y_test_flat, y_pred_flat)
print(f"Test F1 Score: {test_f1_score}")
print(f"Test Accuracy: {test_accuracy}")

with open('sumerian_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(sumerian_tokenizer.to_json())

with open('english_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(english_tokenizer.to_json())

with open('pos_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(pos_tokenizer.to_json())
