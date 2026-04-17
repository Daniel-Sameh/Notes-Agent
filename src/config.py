from dotenv import load_dotenv
import os

class Settings:
    def __init__(self):
        load_dotenv()
        self.groq_api = os.getenv("GROQ_API_KEY")
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///./notes.db")
        self.llm_provider = os.getenv("LLM_PROVIDER", "groq")
    
    def get_llm_api_key(self):
        return self.groq_api


settings = Settings()