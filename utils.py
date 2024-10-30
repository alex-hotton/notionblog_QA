import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def format_metadata(metadata: dict) -> dict:
    """
    Transform a metadata dictionary by flattening nested structures into string values.
    """
    formatted_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, dict):
            non_none_values = {k: v for k, v in value.items() if v is not None}
            formatted_metadata[key] = ', '.join(f"{k}: {v}" for k, v in non_none_values.items()) if non_none_values else ""
        elif isinstance(value, list):
            formatted_metadata[key] = ', '.join(str(item) for item in value)
        elif value is not None:
            formatted_metadata[key] = str(value)
        else:
            formatted_metadata[key] = ""
            
    return formatted_metadata

def load_prompt(filename: str) -> str:
    """
    Load a prompt template from a file.
    """
    if not filename.endswith('.sysprompt'):
        filename += '.sysprompt'
    
    prompt_path = os.path.join('prompts', filename)
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

def init_environment():
    """Initialize environment variables and return common model instances"""
    load_dotenv()
    logger.info("Loading environment variables and initializing models")
    
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    
    try:
        openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        logger.info("Successfully initialized OpenAI models")
        return openai_embeddings, llm
    except Exception as e:
        logger.error(f"Error initializing OpenAI models: {str(e)}")
        raise