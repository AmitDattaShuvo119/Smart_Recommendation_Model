from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved sentiment analysis model and tokenizer
model_path = './model_directory'
tokenizer_path = './tokenizer_directory'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model.eval()

# Define weights for number of reviews and average positivity
reviews_weight = 0.7
positivity_weight = 0.3


def get_external_data(provider_id, category):
    api_url = f"http://127.0.0.1:5000/providers/api/v2/{provider_id}/{category}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f"Failed to fetch data. Status code: {response.status_code}"}


def generate_content_based_recommendation(user_profile, providers_data):
    # Extracting vectors for user profile and each professional
    user_vector = np.array([
        user_profile["user_rating"],
        user_profile["positiveness"],
        user_profile["user_hireCount"],
        user_profile["user_respondTime"]
    ])
    professionals_vectors = np.array([
        [prof["user_rating"], prof["positiveness"], prof["user_hireCount"], prof["user_respondTime"]]
        for prof in providers_data
    ])

    # Calculate cosine similarity between user profile and each professional
    similarity_scores = cosine_similarity([user_vector], professionals_vectors)[0]

    # Filter out professionals with a rating below 4
    qualified_professionals = [
        (prof, sim) for prof, sim in zip(providers_data, similarity_scores) if prof["user_rating"] >= 4
    ]

    # Sort professionals by similarity scores
    sorted_providers = sorted(qualified_professionals, key=lambda x: x[1], reverse=True)

    if sorted_providers:
        # Extract recommended provider
        recommended_provider = sorted_providers[0][0]['_id']
        return recommended_provider
    else:
        return None


def perform_sentiment_analysis(sentences):
    results = []
    total_positiveness = 0
    total_sentences = len(sentences)

    for i, sentence in enumerate(sentences, 1):
        text = sentence.get('text', '')

        if text:
            # Tokenize the input text
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

            # Make predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).tolist()

            # Determine sentiment
            sentiment = 'positive' if preds[0] == 1 else 'negative'

            # Update total positivity
            total_positiveness += 1 if sentiment == 'positive' else 0

            # Append result for the current sentence
            results.append(f"Sentence {i}: {sentiment}")

    # Calculate average positivity
    average_positivity = total_positiveness / total_sentences if total_sentences > 0 else 0

    # Calculate overall score
    overall_score = (reviews_weight * total_sentences) + (positivity_weight * average_positivity)

    # Prepare the final response
    response = {
        'results': results,
        'total_sentences': total_sentences,
        'average_positivity': average_positivity,
        'overall_score': overall_score
    }

    return response


@app.route('/')
def landing_page():
    return 'Welcome to My Flask App'


@app.route('/api/v1/hello', methods=['GET'])
def hello():
    provider_id = request.args.get('id')
    category = request.args.get('category')

    # Hardcoded user feedback for testing
    user_feedback = {
        "user_rating": 5,
        "positiveness": 1,  # Adjust this value based on your requirements
        "user_hireCount": 100,
        "user_respondTime": 2
    }

    external_data = get_external_data(provider_id, category)
    recommendation = generate_content_based_recommendation(user_feedback, external_data)

    return jsonify({'recommendation': recommendation})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sentences = data.get('sentences', [])

        # Perform sentiment analysis
        sentiment_analysis_result = perform_sentiment_analysis(sentences)

        return jsonify(sentiment_analysis_result)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
