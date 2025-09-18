# 📘 RAG Tutor 
RAG Tutor  – Book-based AI Learning Assistant

## Proposed Solution   
A **Retrieval-Augmented Generation (RAG)** system that:  

- Accepts **one or multiple PDF uploads**  
- Splits them into chunks + embeddings  
- Stores them in **ChromaDB**  
- Answers **only from the uploaded books**, with **page citations**  
- Refuses when there is **insufficient evidence**
   
## Tech Stack  

- **Language** → Python 3  
- **Core Libraries & Tools**:  
  - `pypdf` → Extract text from PDFs  
  - `sentence-transformers` → Convert text into embeddings  
  - `chromadb` → Vector database for storage & retrieval  
  - `streamlit` → User-friendly web frontend  
  - `google-generativeai` → Gemini API for response generation  
  - `dotenv` → Manage API keys securely
 
    
## Rules:  
- Always cite the book (e.g., [Page 12]).  
- Use no outside knowledge.  
- Be concise and factual.  
- Refuse if similarity score is too low.

 
 ## 🌐 Deployment  

The app is deployed on **Streamlit Cloud** and can be accessed at:https://ayesha-rag-tutor.streamlit.app/






  
