from flask import Flask, request, jsonify
import joblib
import string
import pandas as pd
from nltk.stem import PorterStemmer
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

CORS(app)

model = joblib.load("svm_spam_model_best.pkl")
vectorizer = joblib.load("vectorizer1.pkl")

df = pd.read_csv("E:\Tee Htwin\contact_data_general_questions.csv")

if not hasattr(model, "coef_"):
    print("Model is not fitted yet!")
else:
    print("Model is fitted!")



# Define a custom stopwords set
stopwords_set = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now"
}
stemmer = PorterStemmer()

def preprocess_message(message):
    """Preprocess a single email message."""
    text = message.lower().translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
    return " ".join(stemmed_words)

@app.route('/')
def home():
    return "Welcome to the Spam Detection API! Use '/overall', '/spam', or '/nonspam' endpoints."

# Overall Messages Route
@app.route('/overall', methods=['GET'])
def overall():
    messages = df[['Message', 'label']].to_dict(orient='records')
    return jsonify({
        "status": "success",
        "total_messages": len(df),
        "data": messages
    })

# Spam Messages Route
@app.route('/spam', methods=['GET'])
def spam():
    spam_messages = df[df['label'] == 1][['Message', 'label']].to_dict(orient='records')
    return jsonify({
        "status": "success",
        "total_spam": len(spam_messages),
        "data": spam_messages
    })

# Non-Spam Messages Route
@app.route('/nonspam', methods=['GET'])
def nonspam():
    non_spam_messages = df[df['label'] == 0][['Message', 'label']].to_dict(orient='records')
    return jsonify({
        "status": "success",
        "total_non_spam": len(non_spam_messages),
        "data": non_spam_messages
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"status": "error", "message": "Message content missing"}), 400

    # Preprocess the input
    processed_message = preprocess_message(data['message'])

    # Vectorize the input
    message_vector = vectorizer.transform([processed_message])
    message_vector = message_vector.toarray()

    # Predict
    prediction = model.predict(message_vector)[0]
    label = "Spam" if prediction == 1 else "Non-Spam"
    return jsonify({"status": "success", "result": label})

if __name__ == '__main__':
    app.run(debug=True)