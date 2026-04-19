from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    database_url: str = "sqlite:///./notes.db"
    chroma_url: str = "./chroma_data"
    llm_provider: str = "groq"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_llm_api_key(self) -> Optional[str]:
        if self.llm_provider == "groq":
            return self.groq_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return None


settings = Settings()