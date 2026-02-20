from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class BlogRequest(BaseModel):
    topic: str
    tone: str = "informative and friendly"
    length: str = "1000-1500"


def build_prompt(topic: str, tone: str, length: str) -> str:
    return f"""Write a complete, SEO-optimized blog post about: "{topic}"

TONE: {tone}
TARGET LENGTH: {length} words

Follow this exact structure:

---
META TITLE: [Write an SEO title 50-60 chars with main keyword]
META DESCRIPTION: [Write a compelling meta description 150-160 chars]
---

# [H1 Main Title - include primary keyword]

[Introduction paragraph - hook the reader, include keyword in first 100 words]

## [H2 - First main section with keyword variation]

[2-3 paragraphs]

## [H2 - Second main section]

[2-3 paragraphs with bullet points if helpful]

## [H2 - Third main section]

[2-3 paragraphs]

## [H2 - Fourth main section or Tips/Best Practices]

[2-3 paragraphs or numbered list]

## Frequently Asked Questions

**Q: [Relevant question 1]**
A: [Clear answer]

**Q: [Relevant question 2]**
A: [Clear answer]

**Q: [Relevant question 3]**
A: [Clear answer]

## Conclusion

[Wrap up with key takeaways and a call to action]

SEO Rules: 1-2% keyword density, use LSI/related terms naturally, short paragraphs, transition words, no keyword stuffing."""


@app.get("/")
def root():
    return {"status": "BlogForge API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.options("/generate")
async def options_generate():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


@app.post("/generate")
async def generate_blog(req: BlogRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    topic = req.topic.strip()
    prefixes = ["write a blog about", "write blog about", "blog about",
                "write about", "blog on", "create a blog about"]
    for p in prefixes:
        if topic.lower().startswith(p):
            topic = topic[len(p):].strip()
            break

    prompt = build_prompt(topic, req.tone, req.length)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional SEO content writer with 10+ years of experience creating blogs that rank on Google."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 3000,
        "temperature": 0.75
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_URL, json=payload, headers=headers)
            data = response.json()
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=data.get("error", {}).get("message", "Groq API error")
                )
            content = data["choices"][0]["message"]["content"]
            return JSONResponse(
                content={"content": content, "topic": topic},
                headers={"Access-Control-Allow-Origin": "*"}
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))