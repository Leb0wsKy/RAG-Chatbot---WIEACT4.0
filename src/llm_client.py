import requests
from src.embedder import cos_sim
from src.scraper import scrape_first_conversations
from src.chunks import chunk_text

def generate_answer(query, relevant_answers):
    # Build the context string from relevant answers
    context = "\n".join([f"{i+1}. {relevant_answers[i]}" for i in range(len(relevant_answers))])
    
    prompt = (
        f"You are an agricultural assistant chatbot helping farmers. "
        f"The farmer has asked the question: '{query}'.\n\n"
        f"You also have some background information from the internet:\n{context}\n\n"
        f"Instructions:\n"
        f"- Your top priority is to give a direct, clear, and practical answer to the farmer’s question.\n"
        f"- If the background information is relevant, you may use it to enrich the answer.\n"
        f"- If it is not relevant, completely ignore it and focus only on the question.\n"
        f"- Do not summarize or retell the background story unless it helps the farmer with their specific question.\n"
        f"- Keep your answer simple, supportive, and easy for a farmer to apply in practice.\n"
        f"- Avoid names, places, or unnecessary storytelling from the background.\n"
        f"- Give practical advice or explanations the farmer can act on.\n\n"
        f"Final Answer for the farmer:"
    )


    
    # Ollama API endpoint (assuming Ollama is running locally on default port)
    url = "http://localhost:11434/api/generate"
    
    # Payload for the API request
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False  # Set to True if you want streaming, but this returns full response
    }
    
    # Make the POST request to Ollama
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        # Extract the generated response
        return response.json()['response'].strip()
    else:
        raise Exception(f"Error calling Ollama API: {response.status_code} - {response.text}") 
