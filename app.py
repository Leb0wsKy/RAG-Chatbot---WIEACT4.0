import streamlit as st
from src.chunks import chunk_text
from src.embedder import cos_sim
from src.scraper import scrape_first_conversations
from src.llm_client import generate_answer  

st.set_page_config(page_title="Farmer Assistant Chatbot", page_icon="🌱", layout="wide")

st.title("🌱 Farmer Assistant Chatbot")
st.write("Ask me questions about your crops, seeds, and farming practices!")

# Input box for the farmer’s query
query = st.text_input("❓ Ask your question:", placeholder="e.g., Why are tomatoes yellow?")

if st.button("Get Answer") and query:
    with st.spinner("Thinking... 🌾"):
        # Step 1: Scrape data dynamically
        res = scrape_first_conversations(query)

        # Step 2: Chunk the scraped data
        chunks = chunk_text(res, chunk_size=500, overlap=50)

        # Step 3: Embed and compute similarity
        emb_res = [cos_sim(query, chunk) for chunk in chunks]

        # Flatten if cos_sim returns lists
        flat_emb_res = [
            item for sublist in emb_res 
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        # Step 4: Generate answer with context
        answer = generate_answer(query, [emb_res_item[0] for emb_res_item in flat_emb_res])

    # Display the answer
    st.success("✅ Answer:")
    st.write(answer)

    # Optional: Show retrieved context for transparency
    with st.expander("🔍 See retrieved context"):
        st.write(res)
