import os
import json
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from prompt_data import DEPARTMENT_DESCRIPTIONS, ABOUT_MELIO, USA_BENEFITS, SYSTEM_PROMPT

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DRIVE_FOLDER_ID = "1c6g3eMKNCuS_fVH8S4JkiMheDx75zNKT"
VISIBLE_DEPARTMENTS = ["Engineering", "Product", "Data", "Design", "Risk", "Growth", "Finance"]

JD_INDEX_FILE = Path(__file__).parent / "jd_index.json"


class JDRequest(BaseModel):
    job_title: str
    department: str
    location: str
    responsibilities: str
    required_skills: str
    bonus_skills: str = ""
    extra_context: str = ""
    reference_jd: str = ""


def _load_jd_index() -> list:
    if JD_INDEX_FILE.exists():
        return json.loads(JD_INDEX_FILE.read_text())
    return []


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/departments")
async def get_departments():
    return [
        {"name": dept, "description": DEPARTMENT_DESCRIPTIONS.get(dept, "")}
        for dept in VISIBLE_DEPARTMENTS
    ]


@app.get("/search-drive")
async def search_drive(query: str):
    """Search the local JD index (pre-loaded from Drive)."""
    jds = _load_jd_index()
    q = query.lower()
    matches = [
        {"id": jd["id"], "title": jd["title"]}
        for jd in jds
        if q in jd["title"].lower() or q in jd.get("content", "").lower()
    ]
    return {"files": matches}


@app.get("/fetch-jd/{file_id}")
async def fetch_jd(file_id: str):
    """Return the content of a JD from the local index."""
    jds = _load_jd_index()
    jd = next((j for j in jds if j["id"] == file_id), None)
    if not jd:
        raise HTTPException(status_code=404, detail="JD not found")
    return {"content": jd["content"]}


@app.post("/generate")
async def generate_jd(req: JDRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    dept_desc = DEPARTMENT_DESCRIPTIONS.get(req.department, "")
    is_usa = req.location.lower() == "usa"

    reference_block = ""
    if req.reference_jd:
        reference_block = f"""
--- REFERENCE JD CHOSEN BY HIRING MANAGER (use as style/structure inspiration) ---
{req.reference_jd[:3000]}
--- END REFERENCE JD ---
"""

    user_prompt = f"""Please write a complete job description for the following role at Melio:

**Job Title:** {req.job_title}
**Department:** {req.department}
**Location:** {"New York / Denver, USA (Hybrid)" if is_usa else "Tel Aviv, Israel (Hybrid)"}

**Key Responsibilities provided:**
{req.responsibilities}

**Required Skills / Qualifications provided:**
{req.required_skills}

**Bonus Skills provided:**
{req.bonus_skills if req.bonus_skills else "Not specified — infer from the role and department"}

**Additional Context:**
{req.extra_context if req.extra_context else "None"}

{reference_block}

---

Use this department description verbatim in the "About the hiring department" section:
{dept_desc}

{"Include the Compensation, Benefits & DEI sections at the end:" if is_usa else "Do NOT include the Compensation/Benefits/DEI sections (this is an Israel-based role)."}
{USA_BENEFITS if is_usa else ""}

Now write the complete job description following Melio's template exactly.
Output only the final JD — no preamble, no commentary.
Use markdown formatting (bold headers with **).
"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit reached — please wait 30 seconds and try again.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)[:200]}")

    return {"jd": message.content[0].text}
