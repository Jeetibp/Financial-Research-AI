"""
OpenAI LLM Client
Wrapper for OpenAI API with structured responses
"""

from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
from src.utils.logger import get_logger
from src.config.config import get_config

logger = get_logger("llm_client")

class LLMClient:
    """OpenAI API client wrapper"""
    
    def __init__(self):
        config = get_config()
        self.client = AsyncOpenAI(api_key=config['openai']['api_key'])
        self.model = config['openai']['model']
        self.temperature = config['openai']['temperature']
        self.max_tokens = config['openai']['max_tokens']
        logger.info(f"LLM Client initialized with model: {self.model}")
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """Generate chat completion"""
        try:
            logger.debug(f"Generating completion with {len(messages)} messages")
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = await self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            logger.debug(f"Completion generated: {len(content)} characters")
            
            return content
            
        except Exception as e:
            logger.error("Error generating completion", error=e)
            raise
    
    async def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        """Generate completion with system prompt"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return await self.generate_completion(messages, **kwargs)
    
    async def extract_structured_data(
        self,
        prompt: str,
        user_input: str
    ) -> Dict[str, Any]:
        """Extract structured JSON data"""
        import json
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = await self.generate_completion(
            messages,
            json_mode=True
        )
        
        return json.loads(response)
    
    def create_research_prompt(self, query: str, context: str = "") -> str:
        """Create research-focused prompt"""
        base_prompt = f"""You are a financial research assistant. 
Analyze the following query and provide detailed, accurate information.

Query: {query}"""
        
        if context:
            base_prompt += f"\n\nContext:\n{context}"
        
        return base_prompt
    
    async def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using LLM with current date context"""
        try:
            from datetime import datetime
            
            # Get current date
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Enhanced system message with date emphasis
            system_message = f"""You are a financial research assistant with access to real-time data.

CRITICAL: Today's date is {current_date}. Always use the most recent data available.

When providing stock prices, market data, or financial information:
- Use TODAY'S date ({current_date}) for all current data
- Clearly state the date of the information
- If data is from the past, explicitly mention it's historical
- Never use outdated information without labeling it as such

Analyze the provided context and answer questions accurately with proper citations."""

            messages = [
                {"role": "system", "content": system_message}
            ]
            
            if context:
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}\n\nIMPORTANT: Today is {current_date}. Provide the most recent data available."
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Question: {prompt}\n\nIMPORTANT: Today is {current_date}."
                })
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
