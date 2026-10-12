"""Small retrieval smoke test for the updated dataset."""
from __future__ import annotations
from config import UNIVERSITIES
from rag import retrieve

QUESTIONS = [
    "What is the Master of Data Science at RMIT?",
    "What accommodation information is available at RMIT?",
    "What accommodation options does the University of Melbourne provide?",
    "What accommodation types does Monash list for international students?",
    "Can international students work while studying?",
    "What workplace rights do international students have?",
]

for question in QUESTIONS:
    print("\n" + "=" * 90)
    print(f"QUESTION: {question}")
    print("=" * 90)
    for rank, result in enumerate(retrieve(question, top_k=5), 1):
        m = result.metadata
        print(f"\n[{rank}] distance={result.distance:.4f}")
        print(f"Entity     : {m.get('university', '')}")
        print(f"Category   : {m.get('category', '')}")
        print(f"Authority  : {m.get('authority', '')}")
        print(f"Section    : {m.get('section', '')}")
        print(f"Source     : {m.get('source_file', '')}")
        print(f"URL        : {m.get('source_url', '')}")
        print(f"Text       : {result.text[:450].replace(chr(10), ' ')}...")
