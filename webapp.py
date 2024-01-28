# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
df = pd.read_csv('cyberbullying_tweets.csv')

# Assuming 'tweet_text' is the column with tweet text and 'cyberbullying_type' is the label
train_data, test_data, train_labels, test_labels = train_test_split(
    df['tweet_text'], df['cyberbullying_type'], test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_vectors, train_labels)

# Make predictions on the test set
predictions = svm_classifier.predict(test_vectors)

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)

print(f'Accuracy: {accuracy}')
print('\nClassification Report:\n', report)

# Now you can use this trained model to predict cyberbullying type for new tweets
# For example:
new_tweet = ["Your new product is amazing!"]
new_tweet_vector = vectorizer.transform(new_tweet)
prediction = svm_classifier.predict(new_tweet_vector)
print(f'Predicted Cyberbullying Type: {prediction[0]}')
