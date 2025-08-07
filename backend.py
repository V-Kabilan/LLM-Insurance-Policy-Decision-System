# backend.py

import fitz
import textwrap
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import requests
import json
import re

TOGETHER_API_KEY = "tgp_v1_qFkMd-yyVEh3BrfpJR0Ji5Ow5-F0msG3iEdTbq_cSm8"

MAX_CHARS_PER_CHUNK = 3000

def detect_policy_type(text):
    text = text.lower()

    score = {
        "Auto Insurance": sum(word in text for word in ["vehicle", "collision", "driver", "motor", "road", "accident"]),
        "Home Insurance": sum(word in text for word in ["property", "home", "fire", "theft", "burglary", "flood", "natural disaster"]),
        "Employee Leave Policy": sum(word in text for word in ["leave", "fmla", "maternity", "absence", "paid leave", "unpaid leave", "holiday"]),
        "Health Insurance": sum(word in text for word in ["hospitalization", "treatment", "claim", "diagnosis", "illness", "disease", "doctor", "surgery"])
    }

    best_match = max(score, key=score.get)
    if score[best_match] == 0:
        return "Unknown"
    return best_match



def extract_chunks_with_metadata(pdf_path, max_chars=MAX_CHARS_PER_CHUNK):
    doc = fitz.open(pdf_path)
    chunks_with_meta = []
    clause_id = 1

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if not page_text.strip():
            continue
        page_chunks = textwrap.wrap(page_text, width=max_chars, break_long_words=False, break_on_hyphens=False)
        for chunk in page_chunks:
            chunks_with_meta.append({
                "id": clause_id,
                "page": page_number,
                "text": chunk.strip()
            })
            clause_id += 1

    doc.close()
    return chunks_with_meta

def build_faiss_index(chunks_with_meta):
    model = SentenceTransformer("models/all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks_with_meta]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.index")
    with open("chunk_texts.pkl", "wb") as f:
        pickle.dump(chunks_with_meta, f)

def search_faiss_index(query, model_path="models/all-MiniLM-L6-v2", top_k=10):
    model = SentenceTransformer(model_path)
    index = faiss.read_index("faiss_index.index")
    with open("chunk_texts.pkl", "rb") as f:
        chunks_with_meta = pickle.load(f)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    # Convert distances to cosine similarity-like scores
    # L2 distance to similarity: similarity = 1 / (1 + distance)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(chunks_with_meta):
            similarity_score = 1 / (1 + dist)
            chunk = chunks_with_meta[i]
            chunk["similarity"] = similarity_score
            results.append(chunk)

    return results

# def query_llm_with_together(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.3, max_tokens=300):
#     url = "https://api.together.xyz/inference"
#     headers = {
#         "Authorization": f"Bearer {TOGETHER_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "temperature": temperature,
#         "max_tokens": max_tokens
#     }
#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         try:
#             return response.json()["choices"][0]["text"].strip()
#         except (KeyError, IndexError):
#             return "⚠️ No valid response from the model."
#     else:
#         return f"⚠️ Request failed with status code {response.status_code}: {response.text}"
def query_llm_with_together(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.7, max_tokens=500):
    import time
    import requests

    url = "https://api.together.xyz/inference"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(3):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["text"].strip()
            except (KeyError, IndexError):
                return "⚠️ No valid response from the model."
        elif response.status_code == 429:
            print("⚠️ Rate limited. Waiting and retrying...")
            time.sleep(2)  # Wait 2 seconds before retry
        else:
            return f"⚠️ Request failed with status code {response.status_code}: {response.text}"

    return "⚠️ LLM failed after multiple retries."



def classify_policy_type(text, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    prompt = f"""
You are an AI that classifies the type of an insurance or HR policy document based on its content.

Document:
{text[:3000]}  # Limit to 3000 characters to stay lightweight

What type of policy is this? Choose only one label from the list below and return just the label, nothing else.

Labels:
- Employee Leave Policy
- Auto Insurance
- Health Insurance
- Home Insurance
- Unknown
"""
    response = query_llm_with_together(prompt, model=model, max_tokens=10)
    return response.strip()


# def extract_first_json(text):
#     import json
#     import re

#     # Clean any markdown formatting (e.g., ```json)
#     cleaned = re.sub(r"```(?:json)?", "", text).strip()

#     # Find all JSON-like objects
#     matches = re.findall(r"{[\s\S]*?}", cleaned)

#     for match in matches:
#         try:
#             return json.loads(match)
#         except json.JSONDecodeError:
#             continue  # Try next block

#     print("❌ No valid JSON found.")
#     return None
def extract_first_json(text):
    import json
    import re

    # Guard against extremely long or obviously malformed output
    if len(text) > 2000:
        print("⚠️ LLM response too long or possibly malformed.")
        return None

    # Clean markdown code fences (```json, ``` etc.)
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    # Find all JSON-like blocks
    matches = re.findall(r"{[\s\S]*?}", cleaned)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue  # Try the next one if current fails

    print("❌ No valid JSON found.")
    return None




def highlight_keywords(text, query):
    """
    Bold all keywords from the query inside the clause text.
    """
    keywords = query.lower().split()
    for word in keywords:
        text = re.sub(f"(?i)({re.escape(word)})", r"**\1**", text)
    return text
