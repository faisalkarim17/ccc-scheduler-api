# main.py  — CCC Scheduler API (M1–M4 consolidated)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx, re

# -----------------------------
# Supabase REST setup
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

# -----------------------------
# App + CORS
# -----------------------------
app = FastAPI(title="CCC Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later: restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers (time + Erlang)
# -----------------------------
DATE_DDMM_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
DATE_ISO_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def parse_date_any(s: str) -> str:
    """Accept dd-mm-yyyy or yyyy-mm-dd; return yyyy-mm-dd."""
    s = s.strip()
    if DATE_ISO_RE.match(s):
        return s
    if DATE_DDMM_RE.match(s):
        d, m, y = s.split("-")
        return f"{y}-{m}-{d}"
    raise HTTPException(400, f"Invalid date format '{s}'. Use dd-mm-yyyy or yyyy-mm-dd.")

def time_to_minutes(tstr: str) -> int:
    hh, mm, ss = tstr.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def erlang_c(a: float, N: int) -> float:
    if N <= a:
        return 1.0
    summ = 0.0
    for k in range(N):
        summ += (a**k) / factorial(k)
    top = (a**N) / factorial(N) * (N / (N - a))
    return top / (summ + top)

def required_agents_for_target(volume: int, aht_sec: int, interval_seconds: int,
                               target_sl: float, target_t: int) -> int:
    if volume <= 0:
        return 0
    lam = volume / interval_seconds
    mu = 1.0 / max(aht_sec, 1)
    a = lam / mu
    N = max(1, ceil(a))
    for _ in range(200):
        if N <= a:
            N = int(ceil(a)) + 1
        Ec = erlang_c(a, N)
        p_wait_gt_t = Ec * exp(-(N * mu - lam) * target_t)
        sl = 1.0 - p_wait_gt_t
        if sl >= target_sl:
            return N
        N += 1
    return N

# -----------------------------
# In-memory CONFIG (admin-editable later via UI)
# -----------------------------
CONFIG: Dict[str, Any] = {
    "interval_seconds": 1800,
    "target_sl": 0.8,
    "target_t": 20,
    "shrinkage": 0.30,
    "shift_minutes": 9 * 60,      # 9h default
    "break_pattern": [15, 30, 15],# minutes
    "break_cap_frac": 0.25,       # at most 25% of assigned can be on break
    "no_head": 60,                # no breaks first hour
    "no_tail": 60,                # no breaks last hour
    "lunch_gap": 120,             # 2h between lunch and 15's
    "site_hours_enforced": False,
    "site_hours": {},             # e.g., {"QA": {"open":"10:00","close":"19:00"}}
    "rest_min_minutes": 12 * 60,  # 12h cross-day rest
    "prev_end_times": {},         # e.g., {"A001":"2025-09-06T19:00:00"}
    "timezone": "UTC",
}

# -----------------------------
# Schemas
# -----------------------------
class RosterRequest(BaseModel):
    date: str
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None

class RangeRequest(BaseModel):
    date_from: str
    date_to: str
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None

class SaveRosterRequest(BaseModel):
    date: str
    roster: List[Dict[str, Any]]

# -----------------------------
# Basic routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "CCC Scheduler API running"}

@app.get("/health/db")
def health_db():
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    try:
        r = httpx.get(
            f"{REST_BASE}/agents",
            headers=HEADERS,
            params={"select": "agent_id", "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        return {"db": "ok", "agents_table": "reachable"}
    except httpx.HTTPError as e:
        raise HTTPException(500, f"Supabase REST error: {e}")

# -----------------------------
# Config endpoints
# -----------------------------
@app.get("/config")
def get_config():
    return CONFIG

@app.put("/config")
def put_config(patch: Dict[str, Any]):
    CONFIG.update(patch or {})
    return CONFIG

# -----------------------------
# Data passthrough (agents / forecasts)
# -----------------------------
@app.get("/agents")
def list_agents(limit: int = 50, offset: int = 0, site_id: Optional[str] = None):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    params = {
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": limit,
        "offset": offset,
        "order": "agent_id.asc",
    }
    if site_id:
        params["site_id"] = f"eq.{site_id}"
    r = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()

@app.get("/forecasts")
def list_forecasts(
    date: str = Query(..., description="dd-mm-yyyy or yyyy-mm-dd"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    date_iso = parse_date_any(date)
    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date_iso}",
        "order": "interval_time.asc,language.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()

# -----------------------------
# Requirements (Erlang C)
# -----------------------------
@app.get("/requirements")
def requirements(
    date: str = Query(..., description="dd-mm-yyyy or yyyy-mm-dd"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    target_sl: Optional[float] = None,
    target_t: Optional[int] = None,
    shrinkage: Optional[float] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    date_iso = parse_date_any(date)

    interval_seconds = interval_seconds or CONFIG["interval_seconds"]
    target_sl = target_sl or CONFIG["target_sl"]
    target_t = target_t or CONFIG["target_t"]
    shrinkage = shrinkage if shrinkage is not None else CONFIG["shrinkage"]

    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date_iso}",
        "order": "interval_time.asc,language.asc,grp.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"

    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    rows = r.json()

    out: List[Dict[str, Any]] = []
    for row in rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req = N_core if shrinkage >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        out.append({
            "date": row["date"],
            "interval_time": row["interval_time"],
            "language": row["language"],
            "grp": row["grp"],
            "service": row["service"],
            "volume": vol,
            "aht_sec": aht,
            "req_core": N_core,
            "req_after_shrinkage": req,
        })
    return out

# -----------------------------
# Roster generation (single day)
# -----------------------------
def _fetch_agents_filtered(language: Optional[str], grp: Optional[str]) -> List[Dict[str, Any]]:
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    a = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params={
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": 2000
    }, timeout=30)
    a.raise_for_status()
    agents = a.json()

    if not language and not grp:
        return agents

    pool = []
    for ag in agents:
        lang_ok = (not language) or (ag["primary_language"] == language) or (ag["secondary_language"] == language)
        grp_ok = (not grp) or (grp in (ag.get("trained_groups") or []))
        if lang_ok and grp_ok:
            pool.append(ag)
    return pool

def _site_open_minutes(site_id: str) -> Optional[tuple]:
    if not CONFIG.get("site_hours_enforced"):
        return None
    sh = CONFIG.get("site_hours", {}).get(site_id)
    if not sh:
        return None
    try:
        o_h, o_m = map(int, sh["open"].split(":"))
        c_h, c_m = map(int, sh["close"].split(":"))
        return (o_h*60 + o_m, c_h*60 + c_m)
    except Exception:
        return None

@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    date_iso = parse_date_any(req.date)
    interval_seconds = req.interval_seconds or CONFIG["interval_seconds"]
    target_sl = req.target_sl or CONFIG["target_sl"]
    target_t = req.target_t or CONFIG["target_t"]
    shrinkage = CONFIG["shrinkage"] if req.shrinkage is None else req.shrinkage

    # forecasts for that date
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date_iso}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
    }, timeout=30)
    r.raise_for_status()
    fc_rows = r.json()

    # compute required per interval (after shrinkage)
    intervals = []
    for row in fc_rows:
        vol, aht = int(row["volume"]), int(row["aht_sec"])
        core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = core if shrinkage >= 0.99 else ceil(core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        intervals.append({"t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after})

    times = sorted({ time_to_minutes(x["t"]) for x in intervals })
    times_map = { t: [iv for iv in intervals if time_to_minutes(iv["t"]) == t] for t in times }
    demand = { t: sum(iv["req"] for iv in times_map[t]) for t in times }
    assigned = { t: 0 for t in times }

    pool = _fetch_agents_filtered(req.language, req.grp)

    SHIFT_MIN = CONFIG["shift_minutes"]
    REST_MIN = CONFIG["rest_min_minutes"]

    roster = []
    used = set()

    prev_end_iso = CONFIG.get("prev_end_times", {}) or {}

    # helper: will shift [start,end) respect site hours and rest?
    def can_place(ag, start_min):
        site = ag.get("site_id") or ""
        # rest rule across days
        last_end = prev_end_iso.get(ag["agent_id"])
        if last_end:
            try:
                last_end_dt = datetime.fromisoformat(last_end)
                start_dt = datetime.fromisoformat(f"{date_iso}T{minutes_to_hhmm(start_min)}:00")
                if (start_dt - last_end_dt).total_seconds() < REST_MIN * 60:
                    return False
            except Exception:
                pass
        # site hours (if enabled)
        rng = _site_open_minutes(site)
        if rng:
            o, c = rng
            if not (o <= start_min and (start_min + SHIFT_MIN) <= c):
                return False
        return True

    def shift_gain(start_min):
        end_min = start_min + SHIFT_MIN
        g = 0
        for t in times:
            if start_min <= t < end_min:
                g += max(demand[t] - assigned[t], 0)
        return g

    # assign greedily at best start slots (30-min grid)
    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    for start_min in candidate_starts:
        end_min = start_min + SHIFT_MIN
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
            pick = None
            for ag in pool:
                if ag["agent_id"] in used:
                    continue
                if not can_place(ag, start_min):
                    continue
                pick = ag
                break
            if not pick:
                break
            used.add(pick["agent_id"])
            roster.append({
                "agent_id": pick["agent_id"],
                "full_name": pick["full_name"],
                "site_id": pick.get("site_id") or "",
                "date": date_iso,
                "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                "notes": "stub assignment",
            })
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1
            if all(assigned[t] >= demand[t] for t in times):
                break

    # --- Break planning with fairness cap ---
    try:
        time_list = list(times)
        def snap(m): return min(time_list, key=lambda x: abs(x - m)) if time_list else m

        cap_frac = CONFIG["break_cap_frac"]
        break_load = {t: 0 for t in time_list}
        def cap_at(t):
            base = assigned[t]
            return 0 if base <= 0 else max(1, int(ceil(base * cap_frac)))
        def span_ok(s, dur):
            e = s + dur
            for t in time_list:
                if s <= t < e:
                    if break_load[t] + 1 > cap_at(t):
                        return False
            return True
        def stress(s, dur):
            e = s + dur
            sc = 0
            for t in time_list:
                if s <= t < e:
                    sc += max(demand[t] - assigned[t], 0)
            return sc
        def choose(cands, dur):
            best = None; best_sc = None
            for s in cands:
                if span_ok(s, dur):
                    sc = stress(s, dur)
                    if best_sc is None or sc < best_sc:
                        best_sc, best = sc, s
            if best is not None:
                return best
            # minimal violation fallback
            best = None; best_pen=None; best_sc=None
            for s in cands:
                e = s + dur; pen=0; sc=0
                for t in time_list:
                    if s <= t < e:
                        over = (break_load[t] + 1) - cap_at(t)
                        if over>0: pen += over
                        sc += max(demand[t] - assigned[t], 0)
                if (best_pen is None or pen < best_pen or (pen == best_pen and (best_sc is None or sc < best_sc))):
                    best_pen, best_sc, best = pen, sc, s
            return best

        NO_HEAD, NO_TAIL, GAP = CONFIG["no_head"], CONFIG["no_tail"], CONFIG["lunch_gap"]
        B1, LUNCH, B3 = CONFIG["break_pattern"][0], CONFIG["break_pattern"][1], CONFIG["break_pattern"][2]

        for item in roster:
            st_str, en_str = [s.strip() for s in item["shift"].split("-")]
            st_h, st_m = map(int, st_str.split(":")); start_min = st_h*60 + st_m
            en_h, en_m = map(int, en_str.split(":")); end_min = en_h*60 + en_m
            if end_min <= start_min: end_min += 24*60

            place_start = start_min + NO_HEAD
            place_end   = end_min   - NO_TAIL
            if place_end - place_start < (B1 + LUNCH + B3 + 2*GAP):
                item["breaks"] = []; continue

            mid = (start_min + end_min)//2
            lunch_target = clamp(mid, place_start + GAP//2, place_end - GAP//2)
            b1_target = clamp(lunch_target - GAP - (B1//2), place_start, place_end)
            b3_target = clamp(lunch_target + GAP + (B3//2), place_start, place_end)

            def window(center, d):
                lo = clamp(center - 60, place_start, max(place_start, place_end - d))
                hi = clamp(center + 60, place_start, max(place_start, place_end - d))
                cands = [t for t in time_list if lo <= t <= hi]
                return cands or [snap(center)]

            # lunch
            Lc = window(lunch_target, LUNCH)
            Ls = choose(Lc, LUNCH) or snap(lunch_target)
            Le = Ls + LUNCH
            for t in time_list:
                if Ls <= t < Le: break_load[t] += 1

            # b1 before lunch
            b1_end_allowed = Ls - GAP
            B1c = [t for t in window(b1_target, B1) if (t + B1) <= b1_end_allowed]
            if not B1c:
                B1c = [t for t in time_list if place_start <= t <= max(place_start, Ls - GAP - B1)]
            B1s = choose(B1c, B1) if B1c else None
            if B1s is None and place_start + B1 <= b1_end_allowed:
                B1s = snap(max(place_start, min(b1_target, b1_end_allowed - B1)))
            B1e = B1s + B1 if B1s is not None else None
            if B1s is not None:
                for t in time_list:
                    if B1s <= t < B1e: break_load[t] += 1

            # b3 after lunch
            b3_start_allowed = Le + GAP
            B3c = [t for t in window(b3_target, B3) if t >= b3_start_allowed]
            if not B3c:
                B3c = [t for t in time_list if min(place_end - B3, b3_target) <= t <= (place_end - B3)]
            B3s = choose(B3c, B3) if B3c else None
            if B3s is None and b3_start_allowed <= (place_end - B3):
                B3s = snap(min(place_end - B3, max(b3_target, b3_start_allowed)))
            B3e = B3s + B3 if B3s is not None else None
            if B3s is not None:
                for t in time_list:
                    if B3s <= t < B3e: break_load[t] += 1

            # attach
            item["breaks"] = []
            if B1s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(B1s % (24*60)),
                                       "end": minutes_to_hhmm(B1e % (24*60)), "kind":"break15"})
            item["breaks"].append({"start": minutes_to_hhmm(Ls % (24*60)),
                                   "end": minutes_to_hhmm(Le % (24*60)), "kind":"lunch30"})
            if B3s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(B3s % (24*60)),
                                       "end": minutes_to_hhmm(B3e % (24*60)), "kind":"break15"})

    except Exception as e:
        print("[break planning] skipped due to error:", e)
        for item in roster:
            item.setdefault("breaks", [])

    summary = {
        "date": date_iso,
        "intervals": [{"time": minutes_to_hhmm(t), "req": demand[t], "assigned": assigned[t],
                       "on_break": 0, "working": assigned[t]} for t in times],
        "agents_used": len(used),
        "roster": roster,
        "notes": [
            "Greedy 9h assignment; fairness cap; site hours enforced (if enabled); "
            "12h cross-day rest via prev_end_times.",
            "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes."
        ],
    }
    return summary

# -----------------------------
# Multi-day range
# -----------------------------
@app.post("/generate-roster-range")
def generate_roster_range(r: RangeRequest):
    date_from_iso = parse_date_any(r.date_from)
    date_to_iso   = parse_date_any(r.date_to)

    d0 = datetime.fromisoformat(f"{date_from_iso}T00:00:00")
    d1 = datetime.fromisoformat(f"{date_to_iso}T00:00:00")
    if d1 < d0:
        raise HTTPException(400, "date_to must be >= date_from")

    days = (d1 - d0).days + 1
    out: Dict[str, Any] = {}
    for i in range(days):
        dd = d0 + timedelta(days=i)
        payload = RosterRequest(
            date=dd.date().isoformat(),
            language=r.language, grp=r.grp,
            shrinkage=r.shrinkage, interval_seconds=r.interval_seconds,
            target_sl=r.target_sl, target_t=r.target_t
        )
        out[payload.date] = generate_roster(payload)
    return out

# -----------------------------
# Save roster (stub – optional)
# -----------------------------
@app.post("/save-roster")
def save_roster(req: SaveRosterRequest):
    date_iso = parse_date_any(req.date)
    # Optional: write to Supabase (requires a table). For now, just ack.
    return {"status": "ok", "inserted": len(req.roster)}

# -----------------------------
# Playground (HTML)
# -----------------------------
PLAYGROUND_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CCC Scheduler — Playground</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1024px;margin:24px auto;padding:0 16px;color:#222}
    h2{margin:16px 0 12px}
    label{display:block;margin:6px 0 2px}
    input,button{padding:8px 10px;margin:6px 8px 12px 0}
    .row{display:flex;gap:12px;flex-wrap:wrap}
    .row > div{flex:1 1 220px;min-width:220px}
    pre{background:#0b0f19;color:#e6edf3;padding:12px;border-radius:8px;overflow:auto}
    button{background:#1060ff;border:0;color:#fff;border-radius:6px;cursor:pointer}
    button:hover{opacity:.92}
  </style>
</head>
<body>
  <h2>CCC Scheduler — Playground</h2>

  <div class="row">
    <div>
      <label>Date (dd-mm-yyyy)</label>
      <input id="date" value="07-09-2025"/>
    </div>
    <div>
      <label>Language</label>
      <input id="lang" value="EN"/>
    </div>
    <div>
      <label>Group</label>
      <input id="grp" value="G1"/>
    </div>
    <div>
      <label>Shrinkage</label>
      <input id="shr" value="0.30"/>
    </div>
    <div>
      <label>Target SL</label>
      <input id="tsl" value="0.8"/>
    </div>
    <div>
      <label>Target T (sec)</label>
      <input id="tt" value="20"/>
    </div>
  </div>

  <div class="row">
    <div>
      <label>Range: From (dd-mm-yyyy)</label>
      <input id="from" value="07-09-2025"/>
    </div>
    <div>
      <label>Range: To (dd-mm-yyyy)</label>
      <input id="to" value="08-09-2025"/>
    </div>
  </div>

  <div class="row">
    <button onclick="runReq()">Run /requirements</button>
    <button onclick="runRoster()">Run /generate-roster</button>
    <button onclick="runRange()">Run /generate-roster-range</button>
    <button onclick="out.textContent=''">Clear output</button>
  </div>

  <pre id="out"></pre>

  <script>
    const out = document.getElementById('out');
    const ddmmyy_to_iso = s => { const [d,m,y] = s.trim().split('-'); return `${y}-${m}-${d}`; };
    const show = x => out.textContent = JSON.stringify(x,null,2);
    function vals(){
      return {
        date: ddmmyy_to_iso(document.getElementById('date').value),
        language: document.getElementById('lang').value,
        grp: document.getElementById('grp').value,
        shrinkage: parseFloat(document.getElementById('shr').value),
        target_sl: parseFloat(document.getElementById('tsl').value),
        target_t: parseInt(document.getElementById('tt').value,10),
      };
    }
    function runReq(){
      const v = vals();
      const q = new URLSearchParams({
        date: v.date, language: v.language, grp: v.grp,
        shrinkage: v.shrinkage, target_sl: v.target_sl, target_t: v.target_t
      });
      fetch('/requirements?'+q).then(r=>r.json()).then(show).catch(e=>show({error:String(e)}));
    }
    function runRoster(){
      const v = vals();
      fetch('/generate-roster',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)})
        .then(r=>r.json()).then(show).catch(e=>show({error:String(e)}));
    }
    function runRange(){
      const v = vals();
      const from = ddmmyy_to_iso(document.getElementById('from').value);
      const to   = ddmmyy_to_iso(document.getElementById('to').value);
      fetch('/generate-roster-range',{method:'POST',headers:{'Content-Type':'application/json'},
        body: JSON.stringify({date_from:from, date_to:to, language:v.language, grp:v.grp,
          shrinkage:v.shrinkage, target_sl:v.target_sl, target_t:v.target_t})})
        .then(r=>r.json()).then(show).catch(e=>show({error:String(e)}));
    }
  </script>
</body>
</html>
"""

@app.get("/playground", response_class=HTMLResponse)
def playground():
    return PLAYGROUND_HTML
