# main.py  — CCC Scheduler API (Supabase-backed, dd-mm-yyyy I/O)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx

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
    HEADERS = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }

# -----------------------------
# App + CORS
# -----------------------------
app = FastAPI(title="CCC Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers — dates & time
# -----------------------------
def parse_ddmmyyyy(d: str) -> str:
    """Input dd-mm-yyyy -> output yyyy-mm-dd (ISO, for DB)."""
    try:
        return datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, f"Invalid date '{d}'. Use dd-mm-yyyy.")

def to_ddmmyyyy(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%m-%Y")

def norm_hhmm(hhmm_or_hhmmss: str) -> str:
    """Return HH:MM from HH:MM or HH:MM:SS."""
    parts = hhmm_or_hhmmss.split(":")
    if len(parts) >= 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    # fallback
    return hhmm_or_hhmmss[:5]

def hhmm_to_minutes(h: str) -> int:
    """
    Accepts 'HH:MM' or 'HH:MM:SS'. Returns minutes since 00:00.
    """
    parts = h.split(":")
    if len(parts) < 2:
        raise HTTPException(400, f"Invalid time '{h}'. Expected HH:MM or HH:MM:SS.")
    hh = int(parts[0]); mm = int(parts[1])
    return hh * 60 + mm

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# -----------------------------
# Config (mutable at runtime)
# -----------------------------
CONFIG: Dict[str, Any] = {
    "interval_seconds": 1800,
    "target_sl": 0.80,
    "target_t": 20,
    "shrinkage": 0.30,

    "shift_minutes": 9 * 60,          # default 9h
    "break_pattern": [15, 30, 15],     # minutes
    "break_cap_frac": 0.25,            # <= 25% of assigned on break concurrently
    "no_head": 60,
    "no_tail": 60,
    "lunch_gap": 120,

    "site_hours_enforced": False,
    "site_hours": {},                  # e.g., {"QA":{"open":"10:00","close":"19:00"}}
    "rest_min_minutes": 12 * 60,
    "prev_end_times": {},              # {agent_id : "YYYY-MM-DDTHH:MM:SS"}
    "timezone": "UTC",
}

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
            timeout=10
        )
        r.raise_for_status()
        return {"db": "ok", "agents_table": "reachable"}
    except httpx.HTTPError as e:
        raise HTTPException(500, f"Supabase REST error: {e}")

# -----------------------------
# Erlang C & Requirement math
# -----------------------------
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
# Thin “list” passthroughs
# -----------------------------
@app.get("/agents")
def list_agents(limit: int = 100, offset: int = 0, site_id: Optional[str] = None):
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
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    iso = parse_ddmmyyyy(date)
    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
        "order": "interval_time.asc,language.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    rows = r.json()
    for row in rows:
        row["date"] = to_ddmmyyyy(row["date"])
        row["interval_time"] = norm_hhmm(row["interval_time"])
    return rows

# -----------------------------
# Requirements (per-interval)
# -----------------------------
@app.get("/requirements")
def requirements(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    interval_seconds: int = CONFIG["interval_seconds"],
    target_sl: float = CONFIG["target_sl"],
    target_t: int = CONFIG["target_t"],
    shrinkage: float = CONFIG["shrinkage"],
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    iso = parse_ddmmyyyy(date)
    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
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
        req_after = N_core if shrinkage >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        out.append({
            "date": to_ddmmyyyy(row["date"]),
            "interval_time": norm_hhmm(row["interval_time"]),
            "language": row["language"],
            "grp": row["grp"],
            "service": row["service"],
            "volume": vol,
            "aht_sec": aht,
            "req_core": N_core,
            "req_after_shrinkage": req_after,
        })
    return out

# -----------------------------
# Body models
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

class SaveRosterRequest(BaseModel):
    date: str
    roster: List[Dict[str, Any]]

# -----------------------------
# Generate roster (single day)
# -----------------------------
@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    interval_seconds = req.interval_seconds or CONFIG["interval_seconds"]
    target_sl = req.target_sl if req.target_sl is not None else CONFIG["target_sl"]
    target_t = req.target_t if req.target_t is not None else CONFIG["target_t"]
    shrinkage = req.shrinkage if req.shrinkage is not None else CONFIG["shrinkage"]

    iso = parse_ddmmyyyy(req.date)

    # forecasts
    f_params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
    }
    fr = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=f_params, timeout=30)
    fr.raise_for_status()
    fc_rows = fr.json()

    intervals = []
    for row in fc_rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = N_core if shrinkage >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        intervals.append({"t": norm_hhmm(row["interval_time"]), "lang": row["language"], "grp": row["grp"], "req": req_after})

    # agents
    ar = httpx.get(
        f"{REST_BASE}/agents",
        headers=HEADERS,
        params={"select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services", "limit": 2000},
        timeout=30
    )
    ar.raise_for_status()
    agents = ar.json()

    if req.language or req.grp:
        pool = [ag for ag in agents if
                (not req.language or (ag["primary_language"] == req.language or ag["secondary_language"] == req.language)) and
                (not req.grp or (req.grp in (ag.get("trained_groups") or [])))]
    else:
        pool = agents[:]

    # greedy assignment
    times = sorted({hhmm_to_minutes(x["t"]) for x in intervals})
    times_map = {t: [iv for iv in intervals if hhmm_to_minutes(iv["t"]) == t] for t in times}
    demand = {t: sum(iv["req"] for iv in times_map[t]) for t in times}
    assigned = {t: 0 for t in times}

    SHIFT_MIN = CONFIG["shift_minutes"]
    roster: List[Dict[str, Any]] = []
    used = set()

    def shift_gain(start_min):
        end_min = start_min + SHIFT_MIN
        return sum(max(demand[t] - assigned[t], 0) for t in times if start_min <= t < end_min)

    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    for start in candidate_starts:
        start_min = start
        end_min = start_min + SHIFT_MIN
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
            pick = None
            for ag in pool:
                if ag["agent_id"] in used:
                    continue
                if CONFIG["site_hours_enforced"]:
                    site = ag.get("site_id")
                    sh = CONFIG["site_hours"].get(site) if site else None
                    if sh:
                        o = hhmm_to_minutes(sh["open"])
                        c = hhmm_to_minutes(sh["close"])
                        if not (o <= start_min and end_min <= c):
                            continue
                prev_end = CONFIG["prev_end_times"].get(ag["agent_id"])
                if prev_end:
                    try:
                        prev_dt = datetime.fromisoformat(prev_end)
                        today_start = datetime.strptime(iso + f"T{minutes_to_hhmm(start_min)}:00", "%Y-%m-%dT%H:%M:%S")
                        if (today_start - prev_dt).total_seconds() < CONFIG["rest_min_minutes"] * 60:
                            continue
                    except Exception:
                        pass
                pick = ag
                break

            if not pick:
                break

            used.add(pick["agent_id"])
            roster.append({
                "agent_id": pick["agent_id"],
                "full_name": pick["full_name"],
                "site_id": pick.get("site_id"),
                "date": req.date,
                "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                "notes": "stub assignment"
            })
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1
            if all(assigned[t] >= demand[t] for t in times):
                break

    # break planning
    try:
        time_list = sorted(times)

        def snap_to_grid(m):
            return min(time_list, key=lambda t: abs(t - m)) if time_list else m

        BREAK_CAP_FRAC = CONFIG["break_cap_frac"]
        break_load = {t: 0 for t in time_list}

        def cap_at(t):
            base = assigned[t]
            if base <= 0:
                return 0
            return max(1, int(ceil(base * BREAK_CAP_FRAC)))

        def span_ok(s, dur):
            e = s + dur
            for t in time_list:
                if s <= t < e and (break_load[t] + 1 > cap_at(t)):
                    return False
            return True

        def stress(s, dur):
            e = s + dur
            sc = 0
            for t in time_list:
                if s <= t < e:
                    sc += max(demand[t] - assigned[t], 0)
            return sc

        def choose_slot(cands, dur):
            best = None; best_sc = None
            for s in cands:
                if span_ok(s, dur):
                    sc = stress(s, dur)
                    if best_sc is None or sc < best_sc:
                        best_sc = sc; best = s
            if best is not None:
                return best
            best = None; best_pen = None; best_sc = None
            for s in cands:
                e = s + dur; pen = 0; sc = 0
                for t in time_list:
                    if s <= t < e:
                        over = (break_load[t] + 1) - cap_at(t)
                        if over > 0: pen += over
                        sc += max(demand[t] - assigned[t], 0)
                if (best_pen is None or pen < best_pen or
                   (pen == best_pen and (best_sc is None or sc < best_sc))):
                    best_pen = pen; best_sc = sc; best = s
            return best

        def window_candidates(center, dur, lo, hi):
            w_lo = clamp(center - 60, lo, max(lo, hi - dur))
            w_hi = clamp(center + 60, lo, max(lo, hi - dur))
            cands = [t for t in time_list if w_lo <= t <= w_hi]
            return cands or [snap_to_grid(center)]

        for item in roster:
            st_str, en_str = [s.strip() for s in item["shift"].split("-")]
            st_h, st_m = map(int, st_str.split(":"))
            en_h, en_m = map(int, en_str.split(":"))
            start_min = st_h*60 + st_m
            end_min   = en_h*60 + en_m
            if end_min <= start_min:
                end_min += 24*60

            NO_HEAD, NO_TAIL, GAP = CONFIG["no_head"], CONFIG["no_tail"], CONFIG["lunch_gap"]
            place_start = start_min + NO_HEAD
            place_end   = end_min   - NO_TAIL
            pattern = CONFIG["break_pattern"]
            if place_end - place_start < (sum(pattern) + 2*GAP):
                item["breaks"] = []
                continue

            mid = (start_min + end_min)//2
            lunch_target = clamp(mid, place_start + GAP//2, place_end - GAP//2)

            lunch_dur = max(10, int(pattern[1]))
            lunch_cands = window_candidates(lunch_target, lunch_dur, place_start, place_end)
            lunch_s = choose_slot(lunch_cands, lunch_dur) or snap_to_grid(lunch_target)
            lunch_e = lunch_s + lunch_dur
            for t in time_list:
                if lunch_s <= t < lunch_e:
                    break_load[t] += 1

            b1_dur = max(5, int(pattern[0]))
            b1_end_allowed = lunch_s - GAP
            b1_cands = [t for t in window_candidates(lunch_s - GAP - 45, b1_dur, place_start, place_end)
                        if (t + b1_dur) <= b1_end_allowed]
            b1_s = choose_slot(b1_cands, b1_dur) if b1_cands else None
            b1_e = b1_s + b1_dur if b1_s is not None else None
            if b1_s is not None:
                for t in time_list:
                    if b1_s <= t < b1_e:
                        break_load[t] += 1

            b3_dur = max(5, int(pattern[2]))
            b3_start_allowed = lunch_e + GAP
            b3_cands = [t for t in window_candidates(lunch_s + GAP + 45, b3_dur, place_start, place_end)
                        if t >= b3_start_allowed]
            b3_s = choose_slot(b3_cands, b3_dur) if b3_cands else None
            b3_e = b3_s + b3_dur if b3_s is not None else None
            if b3_s is not None:
                for t in time_list:
                    if b3_s <= t < b3_e:
                        break_load[t] += 1

            item["breaks"] = []
            if b1_s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)),
                                       "end":   minutes_to_hhmm(b1_e % (24*60)), "kind": f"break{b1_dur}"})
            item["breaks"].append({"start": minutes_to_hhmm(lunch_s % (24*60)),
                                   "end":   minutes_to_hhmm(lunch_e % (24*60)), "kind": f"lunch{lunch_dur}"})
            if b3_s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)),
                                       "end":   minutes_to_hhmm(b3_e % (24*60)), "kind": f"break{b3_dur}"})
    except Exception as e:
        print("[break planning] skipped due to error:", e)
        for item in roster:
            item.setdefault("breaks", [])

    summary = {
        "date": req.date,
        "intervals": [{"time": minutes_to_hhmm(t), "req": demand.get(t, 0),
                       "assigned": assigned.get(t, 0),
                       "on_break": 0, "working": assigned.get(t, 0)} for t in times],
        "agents_used": len(used),
        "roster": roster,
        "notes": [
            "Greedy assignment; fairness cap; site hours (optional); cross-day rest via prev_end_times.",
            "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes."
        ],
    }
    return summary

# -----------------------------
# Generate roster for a range
# -----------------------------
@app.post("/generate-roster-range")
def roster_range(req: RangeRequest):
    d0 = parse_ddmmyyyy(req.date_from)
    d1 = parse_ddmmyyyy(req.date_to)
    start = datetime.strptime(d0, "%Y-%m-%d")
    end = datetime.strptime(d1, "%Y-%m-%d")
    if end < start:
        raise HTTPException(400, "date_to before date_from")

    out: Dict[str, Any] = {}
    cur = start
    while cur <= end:
        day_req = RosterRequest(
            date=to_ddmmyyyy(cur.strftime("%Y-%m-%d")),
            language=req.language,
            grp=req.grp
        )
        out[cur.strftime("%Y-%m-%d")] = generate_roster(day_req)
        cur += timedelta(days=1)
    return out

# -----------------------------
# Save roster
# -----------------------------
@app.post("/save-roster")
def save_roster(req: SaveRosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    if not isinstance(req.roster, list):
        raise HTTPException(400, "roster must be a list")

    iso = parse_ddmmyyyy(req.date)
    payload = []
    for item in req.roster:
        payload.append({
            "date": iso,
            "agent_id": item.get("agent_id"),
            "full_name": item.get("full_name"),
            "site_id": item.get("site_id"),
            "shift": item.get("shift"),
            "breaks": item.get("breaks") or [],
            "meta": item,
        })
    if not payload:
        return {"status": "ok", "inserted": 0}

    r = httpx.post(
        f"{REST_BASE}/rosters",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code, r.text)
    return {"status": "ok", "inserted": len(payload)}

# -----------------------------
# Runtime config endpoints
# -----------------------------
@app.get("/config")
def get_config():
    return CONFIG

@app.put("/config")
def put_config(body: Dict[str, Any]):
    CONFIG.update({k: v for k, v in body.items() if k in CONFIG})
    return CONFIG

# -----------------------------
# Playground — dd-mm-yyyy
# -----------------------------
@app.get("/playground", response_class=HTMLResponse)
def playground():
    html = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>CCC Scheduler — Playground</title>
<style>
 body{{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1024px;margin:24px auto;padding:0 16px}}
 label{{display:block;margin:8px 0 4px;color:#333}} input{{padding:8px;font-size:14px;width:100%;max-width:240px}}
 button{{padding:8px 12px;margin-right:8px}} pre{{background:#f6f8fa;color:#111;padding:12px;border-radius:8px;overflow:auto}}
 .row{{display:flex;gap:12px;flex-wrap:wrap}}
</style></head><body>
<h2>CCC Scheduler — Playground</h2>
<div class="row">
  <div><label>Date (dd-mm-yyyy)</label><input id="date" value="{datetime.utcnow().strftime('%d-%m-%Y')}"></div>
  <div><label>Language</label><input id="lang" value="EN"></div>
  <div><label>Group</label><input id="grp"  value="G1"></div>
  <div><label>Shrinkage</label><input id="shr" value="{CONFIG['shrinkage']}"></div>
</div>
<div class="row">
  <div><label>Target SL</label><input id="tsl" value="{CONFIG['target_sl']}"></div>
  <div><label>Target T (sec)</label><input id="tt"  value="{CONFIG['target_t']}"></div>
</div>
<div class="row">
  <div><label>Range: From</label><input id="from" value="{datetime.utcnow().strftime('%d-%m-%Y')}"></div>
  <div><label>Range: To</label><input id="to"   value="{(datetime.utcnow()+timedelta(days=1)).strftime('%d-%m-%Y')}"></div>
</div>
<p>
  <button onclick="runReq()">Run /requirements</button>
  <button onclick="runRoster()">Run /generate-roster</button>
  <button onclick="runRange()">Run /generate-roster-range</button>
  <button onclick="out.textContent=''">Clear output</button>
</p>
<pre id="out"></pre>

<script>
const out = document.getElementById('out');
function vals() {{
  return {{
    date: document.getElementById('date').value,
    language: document.getElementById('lang').value,
    grp: document.getElementById('grp').value,
    shrinkage: parseFloat(document.getElementById('shr').value),
    target_sl: parseFloat(document.getElementById('tsl').value),
    target_t: parseInt(document.getElementById('tt').value,10),
  }};
}}
function runReq() {{
  const v = vals();
  const q = new URLSearchParams({{ 
    date: v.date, language: v.language, grp: v.grp, 
    target_sl: v.target_sl, target_t: v.target_t, shrinkage: v.shrinkage 
  }});
  fetch('/requirements?'+q).then(r=>r.json()).then(j=>out.textContent=JSON.stringify(j,null,2))
    .catch(e=>out.textContent=String(e));
}}
function runRoster() {{
  const v = vals();
  fetch('/generate-roster', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(v) }})
    .then(r=>r.json()).then(j=>out.textContent=JSON.stringify(j,null,2))
    .catch(e=>out.textContent=String(e));
}}
function runRange() {{
  const v = vals();
  const body = {{ date_from: document.getElementById('from').value,
                  date_to: document.getElementById('to').value,
                  language: v.language, grp: v.grp }};
  fetch('/generate-roster-range', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body) }})
    .then(r=>r.json()).then(j=>out.textContent=JSON.stringify(j,null,2))
    .catch(e=>out.textContent=String(e));
}}
</script>
</body></html>
"""
    return HTMLResponse(html)
