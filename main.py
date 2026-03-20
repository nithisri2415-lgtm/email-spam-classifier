import pandas as pd
import re
from collections import Counter

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data['message'] = data['message'].apply(clean)

# Separate spam and ham
spam = data[data['label'] == 'spam']['message']
ham = data[data['label'] == 'ham']['message']

# Count word frequency
spam_words = Counter()
ham_words = Counter()

for msg in spam:
    spam_words.update(msg.split())

for msg in ham:
    ham_words.update(msg.split())

# Prediction function
def predict(msg):
    msg = clean(msg)
    words = msg.split()

    # Remove common words (stopwords)
    stopwords = ['the','is','are','you','how','what','this','that','to','for','a','an','and','in','on','of']
    words = [w for w in words if w not in stopwords]

    spam_score = 0
    ham_score = 0

    for word in words:
        spam_score += spam_words[word] * 3   # 🔥 strong spam weight
        ham_score += ham_words[word]

    # Extra rule (important 🔥)
    spam_keywords = ['free','win','money','prize','offer','cash','urgent']
    for word in words:
        if word in spam_keywords:
            spam_score += 50   # big boost

    total = spam_score + ham_score + 1

    spam_prob = (spam_score / total) * 100
    ham_prob = (ham_score / total) * 100

    if spam_prob > ham_prob:
        return f"SPAM ({spam_prob:.2f}%)"
    else:
        return f"HAM ({ham_prob:.2f}%)"
# User input loop (FINAL PART)
while True:
    msg = input("Enter message: ")
    print("Prediction:", predict(msg))
    