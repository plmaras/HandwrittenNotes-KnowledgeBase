import os
from groq import Groq
import requests
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import base64

loaded = load_dotenv()

print("Loaded:", loaded)


client = Groq(api_key= os.getenv("GroqAPI"))


MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

PROMPT = """Transcribe this handwritten note exactly as written and Strictly follow the markdown format for content.
Rules:
- Preserve structure: line breaks, bullets, headers.
- Format any math equations neatly using LaTeX syntax in Markdowns style be careful ($...$ for inline, $$...$$ for block).
- If a word is illegible or you're unsure, write your best guess inline as [maybe: "word"].
  Do not silently guess -- always flag uncertainty this way.
- If there's a diagram or sketch that carries meaning, describe it briefly as [diagram: ...].
 
After the transcription, add exactly these two lines:
TITLE: <a short 3-6 word title for this note, filesystem-safe, no punctuation>
SUMMARY: <one sentence describing what this note is about>
 
Output only the transcription and those two lines. No preamble, no markdown fences.
"""
 

def parse_response(content :str) :
    title = re.search(r"^TITLE:\s*(.+)$", content, re.MULTILINE)
    summary = re.search(r"^SUMMARY:\s*(.+)$", content, re.MULTILINE)

    cutoff = title.start() if title else len(content)
    transcription = content[:cutoff].strip()

    uncertain_count = len(re.findall(r"\[maybe:", transcription))


    return {
        "transcription": transcription,
        "title": title.group(1).strip() if title else "",
        "summary": summary.group(1).strip() if summary else "",
        "uncertain_count": uncertain_count,
    }



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def transcribe_image (image_path):
   base64_image = encode_image(image_path)
   extesn = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
   completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{extesn};base64,{base64_image}",
                        },
                    },
                ]
            }
        ],
        temperature= 0.2,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,
    )
   
   content = completion.choices[0].message.content

   return parse_response(content)


def slugify(text: str, fallback: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    text = re.sub(r"\s+", "-", text)
    return text[:50] if text else fallback
 


def write_note(image_path: Path, result: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
 
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(result["title"], fallback=image_path.stem)
    note_path = output_dir / f"{date_str}-{slug}.md"
 
    frontmatter = f"""---
    date: {date_str}
    source_image: "{image_path.name}"
    uncertain_words: {result['uncertain_count']}
    summary: "{result['summary'].replace('"', "'")}"
    ---
    ## {result['title'] or image_path.stem}

    {result['transcription']}

    """

    note_path.write_text(frontmatter, encoding="utf-8")
    return note_path
   


def main():
    input_dir = Path("images")
    output_dir = Path("notes")
 
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
 
    manifest = []
    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        try:
            result = transcribe_image(image_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
 
        note_path = write_note(image_path, result, output_dir)
        print(note_path)
        flag = " needs review" if result["uncertain_count"] > 0 else ""
        print(f"  -> {note_path.name}{flag}")
        manifest.append({
            "image": image_path.name,
            "note": note_path.name,
            "uncertain_count": result["uncertain_count"],
        })
 
    Path(output_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    
 
 
if __name__ == "__main__":
    main()
 
