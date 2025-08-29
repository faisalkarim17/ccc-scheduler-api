# main.py — CCC Scheduler API (dd-mm-yyyy dates)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os, httpx, json
from math import factorial, exp, ceil

# -----------------------------
# Helpers for date formatting
# -----------------------------
DATE_FMT = "%d-%m-%Y"   # dd-mm-yyyy
DB_DATE_FMT = "%Y-%m-%d"  # storage

def parse_date_ddmmyyyy(d: str) -> datetime.date:
    return datetime.strptime(d, DATE_FMT).date()

def format_date_ddmmyyyy(dt: datetime.date) -> str:
    return dt.strftime(DATE_FMT)

def db_date(dt: datetime.date) -> str:
    return dt.strftime(DB_DATE_FMT)

def from_db_date(d: str) -> str:
    return datetime.strptime(d, DB_DATE_FMT).strftime(DATE_FMT)

# -----------------------------
# Supabase REST config
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="CCC Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Config defaults (same as before)
# -----------------------------
CONFIG_DEFAULT: Dict[str, Any] = {
    "interval_seconds": 1800,
    "target_sl": 0.80,
    "target_t": 20,
    "shrinkage": 0.30,
    "shift_minutes": 540,
    "break_pattern": [15, 30, 15],
    "break_cap_frac": 0.25,
    "no_head": 60,
    "no_tail": 60,
    "lunch_gap": 120,
    "site_hours_enforced": False,
    "site_hours": {},
    "rest_min_minutes": 720,
    "prev_end_times": {},
    "timezone": "UTC",
}
CONFIG_CACHE = CONFIG_DEFAULT.copy()
SETTINGS_TABLE = "settings"
SETTINGS_ID = "global"

# -----------------------------
# Small helpers (minutes)
# -----------------------------
def time_to_minutes(tstr: str) -> int:
    hh, mm, *_ = tstr.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    return f"{(m//60)%24:02d}:{m%60:02d}"

# -----------------------------
# Health
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "CCC Scheduler API running"}

# -----------------------------
# Erlang C + staffing
# -----------------------------
def erlang_c(a: float, N: int) -> float:
    if N <= a:
        return 1.0
    s = sum((a**k) / factorial(k) for k in range(N))
    top = (a**N) / factorial(N) * (N / (N - a))
    return top / (s + top)

def required_agents(volume, aht_sec, interval_seconds, target_sl, target_t):
    if volume <= 0: return 0
    lam = volume / interval_seconds
    mu = 1.0 / max(aht_sec, 1)
    a = lam / mu
    N = max(1, ceil(a))
    for _ in range(200):
        if N <= a: N = int(ceil(a)) + 1
        Ec = erlang_c(a, N)
        p_wait = Ec * exp(-(N*mu - lam)*target_t)
        sl = 1.0 - p_wait
        if sl >= target_sl: return N
        N += 1
    return N

# -----------------------------
# Requirements endpoint
# -----------------------------
@app.get("/requirements")
def requirements(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    if not REST_BASE: raise HTTPException(500, "No DB")
    dt = parse_date_ddmmyyyy(date)

    params = {"select": "date,interval_time,language,grp,service,volume,aht_sec",
              "date": f"eq.{db_date(dt)}",
              "order": "interval_time.asc"}
    if language: params["language"] = f"eq.{language}"
    if grp: params["grp"] = f"eq.{grp}"
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()

    out = []
    for row in rows:
        vol, aht = int(row["volume"]), int(row["aht_sec"])
        N = required_agents(vol, aht,
                            CONFIG_CACHE["interval_seconds"],
                            CONFIG_CACHE["target_sl"],
                            CONFIG_CACHE["target_t"])
        out.append({
            "date": from_db_date(row["date"]),
            "interval_time": row["interval_time"],
            "language": row["language"],
            "grp": row["grp"],
            "volume": vol,
            "aht_sec": aht,
            "req": N,
        })
    return out

# -----------------------------
# Roster request/response
# -----------------------------
class RosterRequest(BaseModel):
    date: str   # dd-mm-yyyy
    language: Optional[str] = None
    grp: Optional[str] = None

@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    dt = parse_date_ddmmyyyy(req.date)
    # Just dummy output for clarity — your earlier assignment + break logic fits here
    return {
        "date": req.date,
        "agents_used": 3,
        "roster": [
            {"agent_id": "A001", "full_name": "Test Agent",
             "date": req.date, "shift": "09:00 - 18:00",
             "breaks": [{"start":"12:30","end":"13:00","kind":"lunch"}]}
        ]
    }

# -----------------------------
# Multi-day
# -----------------------------
class RangeReq(BaseModel):
    date_from: str
    date_to: str
    language: Optional[str] = None
    grp: Optional[str] = None

@app.post("/generate-roster-range")
def generate_roster_range(req: RangeReq):
    d1 = parse_date_ddmmyyyy(req.date_from)
    d2 = parse_date_ddmmyyyy(req.date_to)
    out = {}
    cur = d1
    while cur <= d2:
        out[format_date_ddmmyyyy(cur)] = {"date": format_date_ddmmyyyy(cur), "roster":[]}
        cur += timedelta(days=1)
    return out

# -----------------------------
# Save + Get rosters
# -----------------------------
@app.get("/rosters")
def get_rosters(date: str):
    dt = parse_date_ddmmyyyy(date)
    # Return saved (here stub)
    return {"date": date, "roster":[]}

# -----------------------------
# Config endpoints
# -----------------------------
@app.get("/config")
def get_config(): return CONFIG_CACHE

@app.put("/config")
def put_config(body: Dict[str, Any]):
    global CONFIG_CACHE
    CONFIG_CACHE.update(body)
    return CONFIG_CACHE

# -----------------------------
# Playground
# -----------------------------
PLAYGROUND_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>CCC Scheduler Playground</title>
<style>body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:24px}</style></head><body>
<h2>CCC Scheduler — Playground</h2>
<label>Date (dd-mm-yyyy)</label><input id="date" value="07-09-2025">
<label>Language</label><input id="lang" value="EN">
<label>Group</label><input id="grp" value="G1">
<div><button onclick="runRoster()">Run /generate-roster</button></div>
<pre id="out"></pre>
<script>
function runRoster(){
  const d=document.getElementById('date').value;
  const l=document.getElementById('lang').value;
  const g=document.getElementById('grp').value;
  fetch('/generate-roster',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({date:d,language:l,grp:g})})
    .then(r=>r.json()).then(v=>out.textContent=JSON.stringify(v,null,2))
}
</script></body></html>"""

@app.get("/playground")
def playground(): return PLAYGROUND_HTML
