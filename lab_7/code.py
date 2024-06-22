text_preprocessing = """
def preprocess_text(text):
    text = contractions.fix(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.strip()
    return text

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    return data

data = load_and_preprocess_data('data/yelp_reviews.csv')
"""

tokenization ="""
# Tokenize and pad sequences
def tokenize_and_pad(texts, max_length=5000):
    tokenizer = Tokenizer(num_words=max_length)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = sequence.pad_sequences(sequences, maxlen=max_length)
    return padded_sequences, tokenizer

X, tokenizer = tokenize_and_pad(data['cleaned_text'])

# Encode labels
def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

y, label_encoder = encode_labels(data['label'])

"""


model = """
# Build the model
def create_model(input_length):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=input_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model(X_train.shape[1])


history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

"""

eval = """
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""
predict = """
def predict_new_data(model, tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = sequence.pad_sequences(sequences, maxlen=X_train.shape[1])
    predictions = model.predict(padded_sequences)
    return predictions

new_texts = [
    "This restaurant has the best food I've ever tasted!",
    "I am absolutely thrilled with the service at this hotel.",
    "The movie was an incredible experience, full of emotions.",
    "I love the ambiance of this café, so cozy and welcoming.",
    "This product exceeded my expectations in every way.",
    "The food at this place was horrible and tasteless.",
    "I am very disappointed with the customer service I received.",
    "The movie was a complete waste of time and money.",
    "This café is too noisy and uncomfortable.",
    "The product did not meet any of my expectations."
]

predictions = predict_new_data(model, tokenizer, new_texts)
print(predictions)

"""
