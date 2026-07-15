A **Gradio-based tool** that fetches a **YouTube video's transcript**, then lets you **generate a summary** or **ask specific questions** about the video's content using a **RAG pipeline** built with **LangChain**, **FAISS**, and **Groq**. Built as a practice excercise on my RAG & Agentic AI learning path.

**Features:**
- Fetch transcripts directly from YouTube video URLs
- Summarize a video's content in one click
- Ask questions about specific parts of the video using semantic search over the transcript

<img width="1917" height="1028" alt="image" src="https://github.com/user-attachments/assets/b5d3e0a1-ea74-4762-bdba-65f047bf4a99" />

**Working:**
- When a user submits a **YouTube URL**, the app extracts the **video ID** and uses **youtube-transcript-api** to pull the **video's captions** (auto-generated or manually created, preferring **manual when available**), formatted with **timestamps**.
- For a **quick summary**, the **full transcript** is passed directly into a **prompt template**, and Groq's hosted **Llama 3.3 70B model** generates a **concise paragraph** summarizing the video's content.
- For **specific questions**, the transcript is first split into **smaller overlapping chunks** using **LangChain's RecursiveTextSplitter**. Each chunk is converted into a **vector embedding** using a **local HuggingFace embedding model** (**all-MiniLM-L6-v2**), and all **chunks are indexed** in a **FAISS vector store**.
- When a **user asks a question**, the app performs a **similarity search** against this index to retrieve the **most relevant chunks (k=7)** of the transcript, then **augments the context to the query** and feeds it to the LLM to generate a grounded answer.
