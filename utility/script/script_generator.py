import json
from utility.config import get_config


def clean_markdown(text):
    """Remove markdown formatting from text to prevent TTS issues."""
    import re
    
    # Remove bold formatting (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove italic formatting (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove code formatting (`text` or ```text```)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove headers (# text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove links [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def generate_script(topic):
    config = get_config()
    client = config.get_llm_client()
    model = config.get_llm_model()
    provider = config.get_llm_provider()
    
    prompt = (
        """You are a seasoned content writer for a YouTube Shorts channel, specializing in facts videos. 
        Your facts shorts are concise, each lasting less than 50 seconds (approximately 140 words). 
        They are incredibly engaging and original. When a user requests a specific type of facts short, you will create it.

        For instance, if the user asks for:
        Weird facts
        You would produce content like this:

        Weird facts you don't know:
        - Bananas are berries, but strawberries aren't.
        - A single cloud can weigh over a million pounds.
        - There's a species of jellyfish that is biologically immortal.
        - Honey never spoils; archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.
        - The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.
        - Octopuses have three hearts and blue blood.

        You are now tasked with creating the best short script based on the user's requested type of 'facts'.

        Keep it brief, highly interesting, and unique.

        Reply with ONLY a single parsable JSON object with the key 'script' — no
        preamble, no markdown fences, no commentary before or after it. Smaller
        local models are more literal about instructions than large ones, so
        this must be the entire response, verbatim:

        {"script": "Here is the script ..."}
        """
    )
    
    if provider == 'gemini':
        content = _call_gemini(client, topic, prompt)
    elif provider == 'ollama':
        content = _call_ollama_native(model, topic, prompt)
    else:
        content = _call_openai_groq(client, model, topic, prompt)
    
    try:
        # Remove any common prefix that might be added by LLMs (content:, content =, content=, content: , etc.)
        text = content
        for prefix in ['content:', 'content =', 'content =', 'content: ', 'content=']:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Try to find complete JSON object or array
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON found in response")
        
        script_text = text[json_start:json_end+1]
        script = json.loads(script_text)["script"]
        script = clean_markdown(script)
        return script
    except Exception as e:
        print(f"Error: {e}")
        raise
    return script


def _call_openai_groq(client, model, topic, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": topic}
        ]
    )
    return response.choices[0].message.content


def _call_ollama_native(model, topic, prompt):
    """Calls Ollama's native /api/chat instead of going through the
    OpenAI-compatible /v1 shim.

    Two things needed to make a small local model produce reliable JSON,
    both verified live against qwen3.5:4b:

    1. `format: "json"` — grammar-constrained decoding, guarantees
       syntactically valid JSON. (The OpenAI-compat layer's equivalent,
       response_format={"type": "json_object"}, did NOT fix this — see 2.)
    2. `think: False` — qwen3.5 is a hybrid-reasoning model that otherwise
       spends its whole generation budget on a hidden "thinking" pass and
       returns an EMPTY final `content` once JSON-grammar mode is also
       forced. Disabling it dropped a call that previously returned '' down
       to well under a second with correct output. The OpenAI-compat /v1
       endpoint does not honor a passthrough `think` field (tested via both
       the openai SDK's `extra_body` and a raw request to /v1/chat/completions
       — both left thinking on) — the native endpoint is the only place this
       flag reliably applies, hence calling it directly here instead of
       reusing the OpenAI-compatible client from get_llm_client().
    """
    import os
    import requests

    base_url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": topic},
            ],
            "format": "json",
            "think": False,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _call_gemini(client, topic, prompt):
    response = client.generate_content(
        contents=[
            {"role": "user", "parts": [{"text": f"{prompt}\n\nTopic: {topic}"}]}
        ],
        generation_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "max_output_tokens": 8192,
        }
    )
    text = response.text
    
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    
    return text.strip()
