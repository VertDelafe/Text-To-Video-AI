import json
import re
from utility.config import get_config
from utility.utils import log_response, LOG_TYPE_GPT

prompt = """# Instructions

Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 2-4 seconds. The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

For example, if the caption is 'The cheetah is the fastest land animal, capable of running at speeds up to 75 mph', the keywords should include 'cheetah running', 'fastest animal', and '75 mph'. Similarly, for 'The Great Wall of China is one of the most iconic landmarks in the world', the keywords should be 'Great Wall of China', 'iconic landmark', and 'China landmark'.

Important Guidelines:

Use only English in your text queries.
Each search string must depict something visual.
The depictions have to be extremely visually concrete, like rainy street, or cat sleeping.
'emotional moment' <= BAD, because it doesn't depict something visually.
'crying child' <= GOOD, because it depicts something visual.
The list must always contain the most relevant and appropriate query searches.
['Car', 'Car driving', 'Car racing', 'Car parked'] <= BAD, because it's 4 strings.
['Fast car'] <= GOOD, because it's 1 string.
['Un chien', 'une voiture rapide', 'une maison rouge'] <= BAD, because the text query is NOT in English.

Note: Your response should be the response only and no extra text or data.
  """

def fix_json(json_str):
    # Replace typographical apostrophes with straight quotes
    json_str = json_str.replace("’", "'")
    # Replace any incorrect quotes (e.g., mixed single and double quotes)
    json_str = json_str.replace("“", "\"").replace("”", "\"").replace("‘", "\"").replace("’", "\"")
    # Add escaping for quotes within the strings
    json_str = json_str.replace('"you didn"t"', '"you didn\'t"')
    return json_str

def _local_fallback_queries(script, captions_timed, end):
    """Keyword-extraction fallback used whenever the LLM can't be trusted to
    produce the exact schema in time. Deterministic, free, and — unlike a
    dead `return None` — always leaves the pipeline with *something* to
    search Pexels for.
    """
    stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "of", "to", "in", "on", "at", "for", "with", "as", "by", "it", "its", "we", "us", "our", "you"}

    fallback = []
    t = 0.0
    while t < end:
        t_next = min(t + 4.0, end)

        # Extract words spoken during this segment
        segment_words = []
        for (w_t1, w_t2), word in captions_timed:
            # Check if word overlaps with segment
            if w_t1 >= t and w_t1 < t_next:
                clean = re.sub(r'[^\w\s]', '', str(word).lower()).strip()
                if clean and clean not in stop_words:
                    segment_words.append(clean)

        # Join up to 4 words to make a visual query
        query_text = " ".join(segment_words[:4])
        if not query_text:
            query_text = "underwater deep ocean" if "ocean" in script.lower() else "stock background video"

        fallback.append([[t, t_next], [query_text, "underwater ocean depth", "marine life abyss"]])
        t = t_next
    return fallback


def getVideoSearchQueriesTimed(script,captions_timed):
    end = captions_timed[-1][0][1]
    max_retries = 3
    retry_count = 0

    try:
        out = [[[0,0],""]]
        while out[-1][0][1] != end:
            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached. Using current result or local fallback.")
                if out == [[[0,0],""]]:
                    # Every attempt failed to produce the expected schema
                    # (observed live with qwen3.5:4b: format:"json" guarantees
                    # syntactically valid JSON but not schema compliance — it
                    # returned three different unrelated JSON shapes across
                    # three retries). This used to `return None` here, which
                    # skipped the local fallback below entirely (no exception
                    # was raised, so the except block never ran) and crashed
                    # generate_video_url() downstream with a bare
                    # "'NoneType' object is not iterable". Fall through to the
                    # same deterministic fallback the except handler uses.
                    return _local_fallback_queries(script, captions_timed, end)
                return out

            content = call_OpenAI(script,captions_timed).replace("'",'"')
            try:
                out = json.loads(content)
            except Exception as e:
                print("JSON parse error, attempting to fix...")
                print(e)
                try:
                    content = fix_json(content.replace("```json", "").replace("```", ""))
                    out = json.loads(content)
                except Exception as e2:
                    print(f"Failed to fix JSON: {e2}")
                    retry_count += 1
                    continue

            if out[-1][0][1] != end:
                retry_count += 1

        return out
    except Exception as e:
        print("error in response, generating local fallback queries:", e)
        return _local_fallback_queries(script, captions_timed, end)

def call_OpenAI(script,captions_timed):
    config = get_config()
    client = config.get_llm_client()
    model = config.get_llm_model()
    provider = config.get_llm_provider()
    
    user_content = """Script: {}
Timed Captions:{}
""".format(script,"".join(map(str,captions_timed)))
    print("Content", user_content)
    
    if provider == 'gemini':
        response = client.generate_content(
            contents=[
                {"role": "user", "parts": [{"text": f"{prompt}\n\n{user_content}"}]}
            ],
            generation_config={
                "temperature":1.0,
                "top_p": 0.9,
                "max_output_tokens": 8192,
            }
        )
        text = response.text.strip()
    elif provider == 'ollama':
        # Same fix as script_generator.py's _call_ollama_native: calling
        # Ollama's OpenAI-compatible /v1 endpoint left a hybrid-reasoning
        # model (qwen3.5:4b) stuck in its hidden "thinking" pass for this
        # prompt in particular — it's much longer than the script prompt
        # (the full script plus every individual timed caption serialized
        # inline) — observed hanging for minutes with no output. Calling
        # Ollama's native /api/chat with think:false fixes it the same way.
        text = _call_ollama_native(model, prompt, user_content).strip()
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
        )
        text = response.choices[0].message.content.strip()
    
    text = re.sub('\s+', ' ', text)
    
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    
    # Remove "content:" prefix if present (Gemini sometimes adds this)
    if text.startswith('content:'):
        text = text[9:].strip()
    
    # Also check for "content =" format
    if text.startswith('content ='):
        text = text[9:].strip()
    
    # Remove markdown code blocks if still present
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        parsed = json.loads(text)
        print("Text", text)
        log_response(LOG_TYPE_GPT,script,text)
        return text
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Original text: {text[:200]}")
        
        # Try to find complete JSON by looking for patterns
        try:
            # Find last complete array or object
            last_bracket = text.rfind(']')
            if last_bracket > 0:
                trimmed = text[:last_bracket+1]
                parsed = json.loads(trimmed)
                print(f"Successfully trimmed JSON to {len(trimmed)} chars")
                log_response(LOG_TYPE_GPT,script,trimmed)
                return trimmed
        except Exception as e2:
            print(f"Trim attempt failed: {e2}")
        
        # Last resort: default fallback structure
        print("Using default fallback structure")
        default_json = '[[[0.16, 5.29], ["default background video", "stock footage", "generic scene"]], [[5.29, 10.29], ["stock video", "background footage", "video content"]]]'
        return default_json

def _call_ollama_native(model, system_prompt, user_content):
    """See utility/script/script_generator.py's _call_ollama_native for why
    this bypasses the OpenAI-compat /v1 endpoint. Ollama's native `format`
    field also accepts plain "json" for any-valid-JSON output (not just
    JSON objects), which is what this caller needs since the expected shape
    here is a top-level array, not an object."""
    import os
    import requests

    base_url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "format": "json",
            "think": False,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def merge_empty_intervals(segments):
    if segments is None:
        print("No background videos available to merge")
        return None
    
    merged = []
    i = 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            # Find consecutive None intervals
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1
            
            # Merge consecutive None intervals with the previous valid URL
            if i > 0:
                prev_interval, prev_url = merged[-1]
                if prev_url is not None and prev_interval[1] == interval[0]:
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    merged.append([interval, prev_url])
            else:
                merged.append([interval, None])
            
            i = j
        else:
            merged.append([interval, url])
            i += 1
    
    return merged
