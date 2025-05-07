import openai
import anthropic
from google import genai
from google.genai import types
from ollama import chat as ollama_chat
import os

class OllamaClient():
    def __init__(self, logger, model):
        self.logger = logger
        self.model = model
    
    def chat(self, messages):
        response = ollama_chat(
            model=self.model,
            messages=messages,
        )
        return response.message.content

class LLMHelper:
    def __init__(self, logger):
        self.logger = logger
        self.temperature = float(os.getenv("temperature", 0.0))
        self.client = self.initialize_client()
    
    def initialize_client(self):
        """Initialize the Language Model based on environment configurations."""
        if os.getenv("use_openai") == "True":
            self.logger.info("Initializing OpenAI conversation.")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY not found in .env file.")
                return None
            return openai.OpenAI(api_key=api_key)
        if os.getenv("use_gemini") == "True":
            self.logger.info("Initializing Gemini conversation.")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("Error: GOOGLE_API_KEY not found in .env file.")
                return None
            return genai.Client(api_key=api_key)
        if os.getenv("use_azure") == "True":
            self.logger.info("Initializing Azure OpenAI conversation.")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_base = os.getenv("AZURE_OPENAI_API_BASE")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15v1")
            if not all([api_key, api_base, deployment_name]):
                print("Error: AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, or AZURE_OPENAI_DEPLOYMENT_NAME not found in .env file.")
                return None
            return openai.AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )
        if os.getenv("use_anthropic") == "True":
            self.logger.info("Initializing Anthropic conversation.")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY not found in .env file.")
                return None
            return anthropic.Anthropic(api_key=api_key)
        if os.getenv("use_ollama") == "True":
            self.logger.info("Initializing Ollama conversation.")
            return OllamaClient(self.logger,os.getenv("ollama_model_name"))
        
        raise ValueError("No LLM backend selected.")

    def clean_reply(self, reply):
        # Remove the code decorators and backticks that ChatGPT returns
        if reply.startswith('```') and reply.endswith('```'):
            # Remove the first line as that contains the ``` decorator with whatever language or syntax it is outputting
            lines = reply.split('\n')
            cleaned_lines = lines[1:-1]
            return '\n'.join(cleaned_lines)
        
        return reply

    def generate_response(self, system_prompt, prompt, history):
        """
        Generate a response from the LLM.
        """
        history_to_use = history.copy()
        if system_prompt is not None:
            if len(history) > 0 and history[0]["role"] != "system":
                history_to_use = [{"role": "system", "content": system_prompt}] + history
            elif len(history) == 0:
                history_to_use = [{"role": "system", "content": system_prompt}]
            elif history[0]["role"] == "system":
                history_to_use[0]["content"] = system_prompt
        
        # Make the thread
        thread = history_to_use + [{"role": "user", "content": prompt}]

        # Generate the response
        if os.getenv("use_openai") == "True" or os.getenv("use_azure") == "True":
            response = self.client.chat.completions.create(
                model=os.getenv("openai_model_name"),
                messages=thread,
                temperature=self.temperature,
            )
            response = response.choices[0].message.content
        if os.getenv("use_gemini") == "True":
            # Gemini requires us to remodel the history
            gemini_thread = []
            for message in thread:
                if message["role"] == "user":
                    gemini_thread.append(types.UserContent(message["content"]))
                elif message["role"] == "assistant":
                    gemini_thread.append(types.ModelContent(message["content"]))

            if system_prompt is not None:
                config = types.GenerateContentConfig(
                    system_instruction=history_to_use[0]["content"],
                    temperature=self.temperature,
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                )
            
            response = self.client.models.generate_content(
                model=os.getenv("gemini_model_name"),
                contents=gemini_thread,
                config=config,
            )
            response = response.text
        if os.getenv("use_anthropic") == "True":
            # Anthropic requires us to remodel the history
            anthropic_thread = []
            for message in thread:
                if message["role"] != "system":
                    new_message = {
                        "role": message["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"]
                            }
                        ]
                    }
                    anthropic_thread.append(new_message)
            
            if system_prompt is not None:
                response = self.client.messages.create(
                    model=os.getenv("anthropic_model_name"),
                    system=history_to_use[0]["content"],
                    temperature=self.temperature,
                    messages=anthropic_thread,
                )
            else:
                response = self.client.messages.create(
                    model=os.getenv("anthropic_model_name"),
                    temperature=self.temperature,
                    messages=anthropic_thread,
                )
            response = response.content
        if os.getenv("use_ollama") == "True":
            response = self.client.chat(thread)
        
        # Debug print the thread and response
        self.logger.debug(f"[LLMHelper] Thread: {thread}")
        self.logger.debug(f"[LLMHelper] Response: {response}")

        return (response, thread)