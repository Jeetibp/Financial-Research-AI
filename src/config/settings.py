from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(extra='allow', env_file='.env')
    
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str

settings = Settings()