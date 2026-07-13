import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv


def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)
    transcript = ""
    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = "\n".join(
                        f"Text: {snippet.text} Start: {snippet.start}"
                        for snippet in t.fetch()
                    )
            else:
                transcript = "\n".join(
                    f"Text: {snippet.text} Start: {snippet.start}"
                    for snippet in t.fetch()
                )
                break

    return transcript if transcript else None

def chunk_transcript(transcript, chunk_size=500, chunk_overlap=20):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(transcript)
    return chunks

def setup_credentials():
    load_dotenv()
    model_id = "llama-3.3-70b-versatile"
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Set it as an environment variable before running.")
    
    return model_id, groq_api_key

def define_parameters():
    return{
        "temperature": 0,
        "max_tokens": 900,
    }

def initialize_groq_llm(model_id, groq_api_key, parameters):
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=model_id,
        api_key=groq_api_key,
        temperature=parameters["temperature"],
        max_tokens=parameters["max_tokens"]
    )
    return llm

def setup_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    :param chunks: List of text chunks
    :param embedding_model: The embeddig model to use
    :return: FAISS index
    """
    from langchain_community.vectorstores import FAISS
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded trasncript using FAISS index.

    :param faiss_index: FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    results = faiss_index.similarity_search(query, k=k)
    return results

def create_summary_prompt():
    """
    Create a prompt template for summarizing a Youtube Video Transcript.
    :return: ChatPromptTemplate object
    """
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
         Instructions:
         1. Summarize the transcript in a single concise paragraph.
         2. Ignore any timestamps in your summary.
         3. Focus on the spoken content (Text) of the video
         Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.
         """),

         ("user", "Please summarize the following YouTube video transcript:\n{transcript}")
    ])
    return prompt

def create_summary_chain(llm, prompt, verbose=True):
    """
    Create a chain for generating summaries using Langchain's pipe syntax.
    :param llm: Language model instance
    :param prompt: ChatPromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: Runnable chain
    """
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt | llm | StrOutputParser()
    if verbose:
        print(f"Chain created: {prompt} -> {llm.model_name if hasattr(llm, 'model_name') else llm} -> StrOutputParser")

    return chain

def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        ChatPromptTemplate: A ChatPromptTemplate object configured for Q&A tasks.
    """
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context} 

    Based on the above context, please answer the following question: 
    Question: {question}
    """
    from langchain_core.prompts import ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("user", qa_template)
    ])
    
    return prompt_template

def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create a chain for question answering using LangChain's pipe syntax.
    :param llm: The language model to use in the chain.
    :param prompt_template: The prompt template to use for structuring inputs to the language model.
    :param verbose: Whether to enable verbose output for the chain.
    :return: Runnable chain 
    """
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt_template | llm | StrOutputParser()

    if verbose:
        print(f"QA chain created: {prompt_template} -> {llm.model_name if hasattr(llm, 'model_name') else llm} -> StrOutputParser")

    return chain

def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.
    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: Runnable
            The question-answering chain to use for generating answers.
        k: int, optional (default=7)
            The number of relevant documents to retrieve.
    Returns:
        str: The generated answer to the user's question.
    """ 
    relevant_docs = perform_similarity_search(faiss_index, question, k=k)
    context_text = "\n".join(doc.page_content for doc in relevant_docs)
    answer = qa_chain.invoke({"context": context_text, "question": question})

    return answer

transcript = ""
current_video_url = ""

def summarize_video(video_url):
    """
    Title: Summarize Video
    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global transcript, current_video_url

    if not video_url:
        return "Please provide a valid YouTube URL.", "No transcript fetched"
    if video_url != current_video_url:
        transcript = get_transcript(video_url)
        current_video_url = video_url
        
    if not transcript:
        return "No transcript available. Please check the video URL and try again.", "Transcript fetch failed"
    
    model_id, groq_api_key = setup_credentials()
    llm = initialize_groq_llm(model_id, groq_api_key, define_parameters())
    summary_prompt = create_summary_prompt()
    summary_chain = create_summary_chain(llm, summary_prompt)
    summary = summary_chain.invoke({"transcript": transcript})
    return summary, "Transcript fetched successfully"

def answer_question(video_url, user_question):
    """
    Title: Answer User's Question
    Description:
    This function retrieves relevant context from the FAISS index based on the user's query 
    and generates an answer using the fetched transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.
    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global transcript, current_video_url

    if not video_url:
        return "Please provide a valid YouTube URL.", "No transcript fetched"
    
    if video_url != current_video_url:
        transcript = get_transcript(video_url)
        current_video_url = video_url
        
    if not transcript:
        return "No transcript available. Please check the video URL and try again.", "Transcript fetch failed"
    
    if not user_question:
        return "Please provide a valid question.", "Transcript ready"
        
    chunks = chunk_transcript(transcript)
    model_id, groq_api_key = setup_credentials()
    llm = initialize_groq_llm(model_id, groq_api_key, define_parameters())
    embedding_model = setup_embedding_model()
    faiss_index = create_faiss_index(chunks, embedding_model)
    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)
    answer = generate_answer(user_question, faiss_index, qa_chain)
    return answer, "Transcript ready"

with gr.Blocks() as interface:
    video_url = gr.Textbox(label="Youtube Video URL", placeholder="Enter the YouTube Video URL")
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    summarize_btn.click(
        summarize_video,
        inputs = video_url,
        outputs = [summary_output, transcript_status]
    )
    question_btn.click(
        answer_question,
        inputs = [video_url, question_input],
        outputs = [answer_output, transcript_status]
    )

port = int(os.environ.get("PORT", 7860))
interface.launch(server_name="0.0.0.0", server_port=port)

