import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('Fake_news_Detection/train.csv')
print(df.head())
print(df.isnull().sum())

print(df.shape)
df = df.dropna()

# Get independent and dependent features
x = df.drop('label', axis=1)
y = df['label']

# Check if dataset is balanced
print(y.value_counts())

# Vocabulary size
vocab_size = 5000

# Copy independent features
messages = x.copy()
print(messages.head())

# Preprocess the title column
messages.reset_index(drop=True, inplace=True)
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', str(messages['title'].iloc[i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print(corpus[:5])

# One-hot encoding
one_hot_repr = [one_hot(words, vocab_size) for words in corpus]
print(one_hot_repr[:5])

# Pad sequences
sent_length = 20
embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

# Create model
embedding_vector_length = 40
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Prepare data for training
x_final = np.array(embedded_docs)
y_final = np.array(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2, random_state=42)

# Train model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save('Fake_news_Detection/model.h5')

# Make predictions
y_pred = model.predict(x_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

# Evaluate model
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))