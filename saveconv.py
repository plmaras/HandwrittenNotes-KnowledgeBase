import os
from groq import Groq
import re
from dotenv import load_dotenv

loaded = load_dotenv()
client = Groq(api_key= os.getenv("GroqAPI"))


GEN_MODEL_NAME = "llama-3.3-70b-versatile" 
SUMMARY_PROMPT = """You summarize a chat conversation between a user and a notes assistant,
so it can be saved as a durable knowledge base entry.
 
Rules:
- Do not just repeat the conversation -- distill it into what was actually learned/decided.
- Be concise. This will be read later without the original conversation for context.
 
Output exactly these labeled sections, nothing else (no preamble, no markdown fences):
TITLE: <5-8 word title, filesystem-safe, no punctuation>
SUMMARY: <2-3 sentence summary of what the conversation covered and concluded>
KEY_POINTS:
- <key point 1>
- <key point 2>
(as many bullet points as genuinely warranted, don't pad)
TAGS: <2-4 short lowercase tags, comma separated>
"""
 

def save_chat(raw):
    content = client.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": raw},
        ],
        temperature=0.3,
        max_completion_tokens=800,
    )
 
    answer_text = content.choices[0].message.content

    title_match = re.search(r"^TITLE:\s*(.+)$", raw, re.MULTILINE)
    summary_match = re.search(r"^SUMMARY:\s*(.+)$", raw, re.MULTILINE)
    tags_match = re.search(r"^TAGS:\s*(.+)$", raw, re.MULTILINE)
    key_points_match = re.search(r"^KEY_POINTS:\s*\n(.*?)(?=^TAGS:|\Z)", raw, re.MULTILINE | re.DOTALL)
 
    return {
        "title": title_match.group(1).strip() if title_match else "conversation",
        "summary": summary_match.group(1).strip() if summary_match else "",
        "key_points": key_points_match.group(1).strip() if key_points_match else "",
        "tags": [t.strip() for t in tags_match.group(1).split(",")] if tags_match else [],
    }
 



 
 
