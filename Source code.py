# EBPL-DS – Chatbot Project: Customer Support Automation

# === 1. Upload and Load Data ===
from google.colab import files
import pandas as pd
import io

print("📁 Upload Conversation.csv (with 'question' and 'answer' columns)")
uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin-1')

# === 2. Preprocessing ===
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return ' '.join(word for word in text.split() if word not in stop_words)

df = df[['question', 'answer']].dropna()
df['clean_question'] = df['question'].apply(clean_text)

# === 3. Plot 1: Question Length Distribution ===
import matplotlib.pyplot as plt
import seaborn as sns

df['question_length'] = df['question'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,6))
sns.histplot(df['question_length'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Question Lengths')
plt.xlabel('Words per Question')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# === 4. Plot 2: WordCloud of Most Frequent Words ===
from collections import Counter
from wordcloud import WordCloud

all_words = ' '.join(df['clean_question'])
word_freq = Counter(all_words.split())

wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Questions')
plt.show()

# === 5. TF-IDF Vectorization for Chatbot ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_question'])

# === 6. Chatbot Function ===
def chatbot_response(user_input):
    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(input_vec, X).flatten()
    best_idx = similarity.argmax()
    if similarity[best_idx] < 0.3:
        return "🤖 Sorry, I didn’t understand that. Could you please rephrase?"
    return df.iloc[best_idx]['answer']

# === 7. Chatbot Test ===
print("\n💬 Chatbot is ready! Sample responses:")
sample_inputs = [
    "Where is my order?",
    "How do I cancel my plan?",
    "Thanks!",
    "Can I get a refund?"
]

for msg in sample_inputs:
    print(f"\n👤 You: {msg}")
    print(f"🤖 Bot: {chatbot_response(msg)}")
