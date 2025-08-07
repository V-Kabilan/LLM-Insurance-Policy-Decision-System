import streamlit as st
import os
import tempfile
import json
from backend import (
    extract_chunks_with_metadata,
    build_faiss_index,
    search_faiss_index,
    query_llm_with_together,
    extract_first_json,
    highlight_keywords,
    detect_policy_type,
    classify_policy_type
)
from prompt_utils import get_few_shot_examples

st.set_page_config(page_title="LLM Policy Decision System", layout="centered")
st.title("📄 LLM Insurance Policy Decision System")

uploaded_file = st.file_uploader("Upload an insurance policy PDF", type=["pdf"])
user_query = st.text_input("Enter your query (e.g. 'knee surgery, 3-month-old policy')")

if st.button("🔍 Analyze"):
    if not uploaded_file or not user_query.strip():
        st.warning("Please upload a PDF and enter a query.")
    else:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(uploaded_file.read())

        # Extract + embed + index
        with st.spinner("📚 Processing PDF and building index..."):
            chunks_with_meta = extract_chunks_with_metadata(tmp_path)
            build_faiss_index(chunks_with_meta)
             # Combine all text to guess policy type
            full_text = " ".join([chunk["text"] for chunk in chunks_with_meta])
            policy_type = classify_policy_type(full_text)  # Use LLM
            #st.info(f"🧾 Detected Policy Type: {policy_type}")
            policy_type = classify_policy_type(full_text)
            print(policy_type)
            if policy_type == "Unknown":
                policy_type = detect_policy_type(full_text)  # Use rule-based fallback
            st.info(f"🧾 Detected Policy Type: {policy_type}")
            if policy_type == "Unknown":
                st.warning("⚠️ Unable to confidently classify the policy type. Results may be less accurate.")

            few_shot_block = get_few_shot_examples(policy_type)

        # Semantic search
        with st.spinner("🔍 Retrieving relevant clauses..."):
            top_chunks = search_faiss_index(user_query, top_k=10)
            if not top_chunks:
                st.error("❌ No relevant clauses found. Please check the policy.")
                st.stop()
            
            with st.expander("🔍 Retrieved Chunks"):
                for c in top_chunks:
                    st.markdown(f"**Clause {c['id']} (Similarity: {round(c['similarity'], 3)})**: {c['text'][:200]}...")



            selected_chunks = "\n".join(
                [f"{chunk['id']}. {chunk['text']}" for chunk in top_chunks]
            )

        # Build prompt for LLM
        prompt = f"""
You are a {policy_type} expert AI assistant. Your job is to return only a JSON object based strictly on the relevant clauses.

Based only on the user's query and the relevant policy clauses, return a JSON decision.

⚠️ Strict Instructions:
- Use only the given clauses.
- Do not hallucinate or fabricate anything.
- Do not write anything before or after the JSON.
- Return only a single JSON object.
- If no clause is relevant, reject with an empty clause_references list.
- Do not use Markdown, code blocks, or explanations.

User Query: "{user_query}"

Relevant Clauses:
{selected_chunks}

Few-shot examples:
{few_shot_block}

Return JSON in the format below ONLY:

{{
  "decision": "Approved" or "Rejected",
  "justification": "Explain using only the clauses above",
  "clause_references": [clause_id_1, clause_id_2]  ← Must match IDs from above
}}
"""



        with st.spinner("🧠 Asking LLM..."):
            response = query_llm_with_together(prompt)
            with st.expander("🧾 Raw LLM Output"):
                st.text(response)
            result = extract_first_json(response)

        # Show result
        if result:
            # 🛡️ Guardrail: Check for invalid clause references
            clause_ids_available = [str(chunk["id"]) for chunk in top_chunks]
            referenced_clauses = [str(cid) for cid in result.get("clause_references", [])]
            invalid_refs = [cid for cid in referenced_clauses if cid not in clause_ids_available]

            if invalid_refs:
                st.warning(f"⚠️ The model referenced clause(s) {invalid_refs} which were not retrieved. The response might be unreliable.")
            
            st.success("✅ Decision JSON")
            st.json(result)

            


            st.subheader("📄 Referenced Clauses")
            for chunk in top_chunks:
                clause_refs = set(str(ref) for ref in result.get("clause_references", []))
                if str(chunk['id']) in clause_refs:
                    st.markdown(f"**Clause {chunk['id']} (Page {chunk['page']})**")
                    highlighted_text = highlight_keywords(chunk['text'], user_query)
                    st.markdown(highlighted_text)
            if not result.get("clause_references"):
                st.info("ℹ️ No matching clauses were found for this query.")


            # Downloadable JSON
            json_data = json.dumps(result, indent=2)
            st.download_button("⬇️ Download JSON", json_data, file_name="decision_result.json")

        else:
            st.error("⚠️ LLM did not return a valid JSON.")
            st.text(response)
