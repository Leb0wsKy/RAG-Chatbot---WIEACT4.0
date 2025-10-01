# app.py
import streamlit as st
from src.scraper import scrape_first_conversations
from src.retriever import build_corpus_from_documents, retrieve_top_k
from src.llm_client import generate_answer_via_openai