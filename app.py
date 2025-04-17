# app.py
from flask import Flask, render_template, request, jsonify
import json
import os
import logging
import requests
import certifi
from convert_csv_to_json import load_initiatives

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
initiatives_data = load_initiatives()
# DHL OpenAI-compatible endpoint config
DHL_API_URL = "https://apihub-sandbox.dhl.com/genai-test/openai/deployments/text-embedding-ada-002-2/embeddings?api-version=2023-05-15"
DHL_API_KEY = os.environ.get("OPENAI_API_KEY")  # Replace with actual key or keep secure

# Get embedding from DHL endpoint
def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": os.environ.get("OPENAI_API_KEY")
    }
    payload = {
        "input": [text]
    }

    response = requests.post(DHL_API_URL, headers=headers, json=payload, verify=certifi.where())
    response.raise_for_status()
    embedding = response.json()["data"][0]["embedding"]
    return embedding

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_reason_from_dhl(user_input, initiative_description):
    DHL_CHAT_API_URL = "https://apihub-sandbox.dhl.com/genai-test/openai/deployments/gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-10-21"

    headers = {
        "Content-Type": "application/json",
        "api-key": os.environ.get("OPENAI_API_KEY")
    }

    # Ensure the API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("API key is missing. Please set the OPENAI_API_KEY environment variable.")

    prompt = f"""You are a helpful assistant. A user entered the following input:

"{user_input}"

You matched it with this initiative:

"{initiative_description}"

Explain in one sentence why this is a relevant match.
"""

    payload = {
        "messages": [
            {"role": "system", "content": "Explain why this campfire is similar to the search criteria. Give me a 2 short sentence description"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        # Send the request to the DHL API
        response = requests.post(DHL_CHAT_API_URL, headers=headers, json=payload, verify=certifi.where())
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the response
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        # Catch all request-related exceptions
        logger.error(f"Error making request to DHL Chat API: {e}")
        return "Error making request."

    except KeyError as e:
        # Catch missing key in the response JSON
        logger.error(f"Error parsing API response: {e}")
        return "Reasoning not available."

    except Exception as e:
        # Catch any other exceptions
        logger.error(f"Unexpected error: {e}")
        return "Reasoning not available."


# Find similar initiatives using embeddings
def find_similar_initiatives(user_description, initiatives, max_results=12):  # Increased to 12
    user_embedding = get_embedding(user_description)

    raw_results = []
    for initiative in initiatives:
        if not isinstance(initiative, dict):
            continue

        initiative_desc = initiative.get('description', '')
        if not initiative_desc:
            continue

        initiative_embedding = initiative.get("embedding")
        if not initiative_embedding:
            continue

        similarity = calculate_cosine_similarity(user_embedding, initiative_embedding)

        raw_results.append({
            'title': initiative.get('title', 'No Title'),
            'owner': initiative.get('owner', 'Unknown'),
            'campfire_id': initiative.get('campfire_id', 'N/A'),
            'link': initiative.get('link', '#'),
            'maturity': initiative.get('maturity', 'Unknown'),
            'similarity': similarity,
            'description': initiative_desc
        })

    top_matches = sorted(raw_results, key=lambda x: x['similarity'], reverse=True)[:max_results]

    for match in top_matches:
        match['reason'] = get_reason_from_dhl(user_description, match['description'])
        del match['description']  # cleanup before returning

    return top_matches


# Cosine similarity calculation
def calculate_cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 * magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_description = request.form.get('description', '')
    print("User input:", user_description)

    similar_initiatives = find_similar_initiatives(user_description, initiatives_data)

    print("Similar initiatives:", similar_initiatives)
    return jsonify(similar_initiatives)

#if __name__ == '__main__':
    #app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)