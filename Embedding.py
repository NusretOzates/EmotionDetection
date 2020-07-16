from tensorflow import keras

data = keras.datasets.imdb
(train_data, train_label), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post',
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=250)

print(len(test_data[0]), len(train_data[0]))


def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])


model = keras.Sequential([
    keras.layers.Embedding(10000, 32),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_label[:10000]
y_train = train_label[10000:]

fitmodel = model.fit(x_train, y_train, batch_size=512, epochs=100, validation_data=(x_val, y_val), verbose=1)

result = model.evaluate(test_data, test_labels)

print(result)
