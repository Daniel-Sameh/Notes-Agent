from ..config import settings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

class LLMProvider:
    """
    A strict Singleton provider for the chosen LLM using lazy initialization.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMProvider, cls).__new__(cls, *args, **kwargs)
            cls._instance._client = None  # Delay initialization
        return cls._instance

    @property
    def client(self) -> BaseChatModel:
        """
        Lazy-loads and returns the LLM client instance.
        """
        if self._client is None:
            self._initialize_client()
        return self._client

    def _initialize_client(self):
        provider = settings.llm_provider
        
        if provider == "groq":
            api_key = settings.get_llm_api_key()
            if not api_key:
                raise ValueError("Groq API key is missing. Please set GROQ_API_KEY in your .env file.")
            
            self._client = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=api_key
            )

        elif provider == "gemini":
            api_key = settings.get_llm_api_key()
            if not api_key:
                raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY in your .env file.")
            
            self._client = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=api_key,
            )

        elif provider == "llama":
            self._client = ChatOllama(
                model="llama3.2:3b",
                temperature=0
            )
        
        elif provider == "openrouter":
            api_key = settings.get_llm_api_key()
            if not api_key:
                raise ValueError("OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your .env file.")
            
            self._client = ChatOpenAI(
                model=settings.llm_model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

# Module-level variable mapping to the singleton instance
llm = LLMProvider()
