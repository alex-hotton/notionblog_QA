import streamlit as st
import uuid
import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import logging
from datetime import datetime

def setup_logging():
    """Configure logging for both console and file output"""
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # File handler
    log_filename = f"chat_logs/chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("chat_logs", exist_ok=True)
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

from langchain_community.document_loaders import NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from utils import format_metadata, load_prompt, init_environment

# Initialize environment and models
openai_embeddings, llm = init_environment()

# Load prompts
contextualize_prompt = load_prompt("contextualize.sysprompt")
answer_prompt = load_prompt("answer.sysprompt")

def create_new_vectorstore(persist_directory: str) -> Chroma:
    """Create a new vectorstore from Notion documents"""
    logger.info("Creating new vectorstore...")
    notion_loader = NotionDBLoader(
        integration_token=os.getenv("NOTION_INTEGRATION_TOKEN"),
        database_id=os.getenv("NOTION_DATABASE_ID"),
    )

    docs = notion_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    for doc in all_splits:
        if doc.metadata:
            doc.metadata = format_metadata(doc.metadata)

    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=openai_embeddings, 
        persist_directory=persist_directory
    )
    logger.info("Vectorstore created and persisted to disk")
    return vectorstore

# Create or load vectorstore
CHROMA_DIR = "./chroma"

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    logger.info("Loading existing vectorstore from disk...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=openai_embeddings
    )
else:
    vectorstore = create_new_vectorstore(CHROMA_DIR)

retriever = vectorstore.as_retriever()

# Set up RAG chain
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# State management
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def call_model(state: State):
    try:
        logger.info("="*50)
        logger.info(f"New query received: {state['input']}")
        logger.info(f"Chat history length: {len(state['chat_history'])}")
        
        response = rag_chain.invoke(state)
        
        logger.info("Retrieved context:")
        logger.info("-"*30)
        logger.info(response['context'])
        logger.info("-"*30)
        logger.info("Generated answer:")
        logger.info(response['answer'])
        logger.info("="*50)
        
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }
    except Exception as e:
        logger.error(f"Error in call_model: {str(e)}")
        raise

# Set up workflow
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Streamlit interface
def main():
    logger.info("Starting new session")

    # Initialize session states
    if "thread_id" not in st.session_state:
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id
        logger.info(f"New conversation started with thread_id: {thread_id}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input and processing
    if prompt := st.chat_input("What would you like to know?"):
        logger.info("\n" + "="*50)
        logger.info(f"User Input: {prompt}")
        logger.info(f"Thread ID: {st.session_state.thread_id}")
        logger.info(f"Current chat history length: {len(st.session_state.chat_history)}")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        input_state = {
            "input": prompt,
            "chat_history": st.session_state.chat_history,
        }
        
        try:
            result = app.invoke(input_state, config=config)
            logger.info("Assistant Response:")
            logger.info(result["answer"])
            
            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=result["answer"])
            ])
            
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            with st.chat_message("assistant"):
                st.write(result["answer"])
                
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

if __name__ == "__main__":
    main() 