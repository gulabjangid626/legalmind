# app/core/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    ENV: str = "development"
    
    # Accept both singular and plural API keys
    GEMINI_API_KEYS: str = Field("", alias="GEMINI_API_KEYS")
    GEMINI_API_KEY: str = Field("", alias="GEMINI_API_KEY")
    PINECONE_API_KEY: str = Field("", alias="PINECONE_API_KEY")
    
    LLM_MODEL: str = "gemini-2.5-flash"
    MAX_CHUNK_TOKENS: int = 1_000

    MAX_FILE_SIZE_MB: int = 10
    CACHE_TTL: int = 3600
    AI_ENHANCEMENT_ENABLED: bool = True
    
    # Add a property to automatically split the keys into a list
    @property
    def GEMINI_KEY_LIST(self) -> List[str]:
        keys = self.GEMINI_API_KEYS or self.GEMINI_API_KEY
        if not keys:
            return []
        return [key.strip() for key in keys.split(',') if key.strip()]

    model_config = SettingsConfigDict(
        # Check current directory and parent directories for environment variables
        env_file=[
            ".env",
            "../.env",
            "../.env.local",
            "../LegalMind/.env",
            "../LegalMind/.env.local"
        ],
        env_prefix='',
        extra='ignore'
    )

settings = Settings()

