import csv
import json
import requests
import certifi
import os
import time

# DHL OpenAI-compatible endpoint config
DHL_API_URL = "https://apihub-sandbox.dhl.com/genai-test/openai/deployments/text-embedding-ada-002-2/embeddings?api-version=2023-05-15"

DHL_API_KEY = os.environ.get("OPENAI_API_KEY")  # Must be set in environment

# Load existing initiatives
def load_initiatives():
    try:
        with open('initiatives.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get('initiatives', [])
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading initiatives: {e}")
        return []

# Batch request for embeddings
def get_embeddings_batch(text_list):
    headers = {
        "Content-Type": "application/json",
        "api-key": os.environ.get("OPENAI_API_KEY")
    }
    payload = {
        "input": text_list
    }

    response = requests.post(DHL_API_URL, headers=headers, json=payload, verify=certifi.where())
    response.raise_for_status()
    result = response.json()
    embeddings = [item['embedding'] for item in result['data']]
    return embeddings

# Load CSV, find new initiatives, and embed them in batches
def load_new_initiatives_from_csv(csv_file='CampfireData.csv', batch_size=10):
    existing_initiatives = load_initiatives()
    existing_ids = {item['campfire_id'] for item in existing_initiatives}

    pending_initiatives = []
    descriptions = []

    with open(csv_file, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            campfire_id = row.get("Campfire_Id", "").strip()
            if not campfire_id or campfire_id in existing_ids:
                continue

            description = row.get("Description", "").strip()
            if not description:
                continue

            initiative = {
                "title": row.get("Title", "").strip(),
                "owner": row.get("Owner", "").strip(),
                "campfire_id": campfire_id,
                "description": description,
                "link": row.get("Link", "").strip(),
                "maturity": row.get("Maturity Level", "").strip()
            }

            pending_initiatives.append(initiative)
            descriptions.append(description)

    # Embed in batches
    print(f"Found {len(pending_initiatives)} new initiatives.")
    embedded_initiatives = []
    for i in range(0, len(descriptions), batch_size):
        desc_batch = descriptions[i:i+batch_size]
        try:
            embeddings = get_embeddings_batch(desc_batch)
        except Exception as e:
            print(f"Batch embedding failed: {e}")
            break  # Optionally retry or skip

        for j, emb in enumerate(embeddings):
            initiative = pending_initiatives[i + j]
            initiative['embedding'] = emb
            embedded_initiatives.append(initiative)

        time.sleep(1)  # To respect rate limits

    return existing_initiatives + embedded_initiatives

# Save to JSON
def save_to_json(initiatives, json_file='initiatives.json'):
    data = {"initiatives": initiatives}
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Saved {len(initiatives)} initiatives to {json_file}")

# Convert process
def convert_csv_to_json():
    updated = load_new_initiatives_from_csv()
    save_to_json(updated)

# Run
convert_csv_to_json()
