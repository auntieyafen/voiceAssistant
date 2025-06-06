from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    APP_NAME: str = "Voice Assistant Backend"
    DEBUG: bool = False
    
    # OpenAI API settings
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    # Google Gemini API settings
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # API Key authentication (former version)
    API_KEY_HEADER: str = "Authorization"
    API_KEY_PREFIX: str = "Bearer"
    API_KEY: str
    
    # Supabase settings
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_JWT_SECRET: str  # Get from supabase
    
    # Database settings - only use DATABASE_URL
    DATABASE_URL: str
    
    # Redis settings (for caching and future message queue)
    REDIS_URL: Optional[str] = None  # Redis connection URL (takes precedence if set)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings() 