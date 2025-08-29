from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Optional, Dict, Any, List, Tuple
from math import factorial, exp, ceil
from datetime import datetime, timedelta
import os, httpx

# ---------- Supabase setup ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS: Dict[str, str] = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

# ---------- App ----------
app = FastAPI(title="CCC Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # limit to your UI origins later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- In-memory CONFIG with sensible defaults ----------
CONFIG: Dict[str, Any] = {
    # staffing targets
    "interval_seconds": 1800,
    "target_sl": 0.80,
    "target_t": 20,
    "shrinkage": 0.30,

    # shift & breaks
    "shift_minutes": 540,               # 9h
    "break_pattern": [15, 30, 15],      # minutes
    "break_cap_frac": 0.25,             # <= 25% of assigned can be on break in any interval
    "no_head": 60,                      # no break in first X minutes of shift
    "no_tail": 60,                      # no break in last  X minutes of shift
    "lunch_gap": 120,                   # min 2h between 15m and 30m (and vice versa)

    # site hours and rest
    "site_hours_enforced": False,
    "site_hours": {},                   # e.g. {"QA":{"open":"10:00","close":"19:00"}}
    "rest_min_minutes": 12*60,          # 12h
    "prev_end_times": {},               # { "A001": "2025-09-06T19:00:00", ... }

    # misc
    "timezone": "UTC",
}

# ---------- Helpers ----------
def erlang_c(a: float, N: int) -> float:
    if N <= a:
        return 1.0
    s = 0.0
    for k in range(N):
        s += (a**k) / factorial(k)
    top = (a**N) / factorial(N) * (N / (N - a))
    return top / (s + top)

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
        p_wait_gt_t = Ec * exp(-(N*mu - lam) * target_t)
        sl = 1.0 - p_wait_gt_t
        if sl >= target_sl:
            return N
        N += 1
    return N

def time_to_minutes(tstr: str) -> int:
    # accepts "HH:MM" or "HH:MM:SS"
    parts = tstr.split(":")
    hh, mm = int(parts[0]), int(parts[1])
    return hh*60 + mm

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def site_open_close_for(agent_site: str) -> Optional[Tuple[int, int]]:
    sh = CONFIG.get("site_hours") or {}
    if not sh:
        return None
    row = sh.get(agent_site)
    if not row:
        return None
    try:
        o = time_to_minutes(row["open"])
        c = time_to_minutes(row["close"])
        return (o, c)
    except Exception:
        return None

def rest_ok(agent_id: str, proposed_start_min: int, date_str: str) -> bool:
    prev = CONFIG.get("prev_end_times") or {}
    if agent_id not in prev:
        return True
    try:
        last_end = datetime.fromisoformat(prev[agent_id])
    except Exception:
        return True
    day = datetime.fromisoformat(date_str)
    proposed_dt = day.replace(hour=proposed_start_min//60, minute=proposed_start_min%60, second=0, microsecond=0)
    need_gap = timedelta(minutes=int(CONFIG.get("rest_min_minutes", 720)))
    return proposed_dt - last_end >= need_gap

# ---------- Routes ----------
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
    date: str = Query(..., description="YYYY-MM-DD"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date}",
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

@app.get("/requirements")
def requirements(
    date: str = Query(..., description="YYYY-MM-DD"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    target_sl: Optional[float] = None,
    target_t: Optional[int] = None,
    shrinkage: Optional[float] = None
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # take from query or fall back to CONFIG
    interval_seconds = interval_seconds or CONFIG["interval_seconds"]
    target_sl = target_sl if target_sl is not None else CONFIG["target_sl"]
    target_t = target_t if target_t is not None else CONFIG["target_t"]
    shrinkage = shrinkage if shrinkage is not None else CONFIG["shrinkage"]

    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date}",
        "order": "interval_time.asc,language.asc,grp.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"

    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()

    out: List[Dict[str, Any]] = []
    for row in rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, float(target_sl), int(target_t))
        if shrinkage >= 0.99:
            req = N_core
        else:
            req = ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
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

# ---------- Roster ----------
from pydantic import BaseModel

class RosterRequest(BaseModel):
    date: str
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None

@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # config values
    cfg = CONFIG
    SHRINK = cfg["shrinkage"] if req.shrinkage is None else req.shrinkage
    ISEC = cfg["interval_seconds"] if req.interval_seconds is None else req.interval_seconds
    TSL  = cfg["target_sl"] if req.target_sl is None else req.target_sl
    TT   = cfg["target_t"] if req.target_t is None else req.target_t
    SHIFT_MIN = int(cfg["shift_minutes"])
    BREAKS = list(cfg["break_pattern"])
    NO_HEAD = int(cfg["no_head"])
    NO_TAIL = int(cfg["no_tail"])
    GAP = int(cfg["lunch_gap"])
    CAP_FRAC = float(cfg["break_cap_frac"])

    # forecasts for that day / lang / grp
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{req.date}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
    }, timeout=30)
    r.raise_for_status()
    fc_rows = r.json()

    # per-interval req
    intervals = []
    for row in fc_rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, ISEC, float(TSL), int(TT))
        req_after = N_core if float(SHRINK) >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(float(SHRINK), 0.95))))
        intervals.append({"t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after})

    # agents
    a = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params={
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": 2000
    }, timeout=30)
    a.raise_for_status()
    agents = a.json()

    # capability filter
    if req.language or req.grp:
        pool = [
            ag for ag in agents
            if (not req.language or (ag["primary_language"] == req.language or ag["secondary_language"] == req.language))
            and (not req.grp or (req.grp in (ag.get("trained_groups") or [])))
        ]
    else:
        pool = agents[:]

    # Site hours and rest prefilters (soft; still validate per-shift)
    # Build time grid
    times = sorted({ time_to_minutes(x["t"]) for x in intervals })
    times_map = { t: [iv for iv in intervals if time_to_minutes(iv["t"])==t] for t in times }
    demand = { t: sum(iv["req"] for iv in times_map[t]) for t in times }
    assigned = { t: 0 for t in times }

    roster: List[Dict[str, Any]] = []
    used: set = set()

    # helper to measure coverage gain
    def shift_gain(start_min: int) -> int:
        end_min = start_min + SHIFT_MIN
        g = 0
        for t in times:
            if start_min <= t < end_min:
                g += max(demand[t] - assigned[t], 0)
        return g

    # candidate starts = every interval mark; sort by gain
    candidate_starts = sorted({t for t in times}, key=lambda s: -shift_gain(s))

    for start_min in candidate_starts:
        end_min = start_min + SHIFT_MIN
        # fill while unmet demand remains in span
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
            # pick a still-free agent that passes site-hours & rest for this shift
            pick = None
            for ag in pool:
                if ag["agent_id"] in used:
                    continue
                # site hours (if enabled)
                if cfg.get("site_hours_enforced"):
                    oc = site_open_close_for(ag.get("site_id") or "")
                    if oc:
                        o, c = oc
                        if not (o <= start_min and end_min <= c):
                            continue
                # rest across days
                if not rest_ok(ag["agent_id"], start_min, req.date):
                    continue
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
                "notes": "stub assignment",
            })
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1

            if all(assigned[t] >= demand[t] for t in times):
                break

    # -------- Break planning with fairness caps --------
    try:
        time_list = list(times)
        break_load = {t: 0 for t in time_list}

        def cap_at(t):
            base = assigned[t]
            if base <= 0:
                return 0
            return max(1, int(ceil(base * CAP_FRAC)))

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

        def choose_slot(cands, dur):
            # pass 1: feasible under cap
            best = None; best_sc = None
            for s in cands:
                if span_ok(s, dur):
                    sc = stress(s, dur)
                    if best_sc is None or sc < best_sc:
                        best_sc = sc; best = s
            if best is not None:
                return best
            # pass 2: minimal violation
            best = None; best_pen = None; best_sc = None
            for s in cands:
                e = s + dur; pen = 0; sc = 0
                for t in time_list:
                    if s <= t < e:
                        over = (break_load[t] + 1) - cap_at(t)
                        if over > 0: pen += over
                        sc += max(demand[t] - assigned[t], 0)
                if (best_pen is None or pen < best_pen
                    or (pen == best_pen and (best_sc is None or sc < best_sc))):
                    best = s; best_pen = pen; best_sc = sc
            return best

        def win(center, dur, lo, hi):
            w_lo = clamp(center - 60, lo, max(lo, hi - dur))
            w_hi = clamp(center + 60, lo, max(lo, hi - dur))
            cands = [t for t in time_list if w_lo <= t <= w_hi]
            return cands or [min(time_list, key=lambda t: abs(t-center))]

        for item in roster:
            st_str, en_str = [s.strip() for s in item["shift"].split("-")]
            st_h, st_m = map(int, st_str.split(":"))
            en_h, en_m = map(int, en_str.split(":"))
            start_min = st_h*60 + st_m
            end_min = en_h*60 + en_m
            if end_min <= start_min:
                end_min += 24*60

            place_start = start_min + NO_HEAD
            place_end   = end_min   - NO_TAIL
            if place_end - place_start < (sum(BREAKS) + 2*GAP):
                item["breaks"] = []
                continue

            # anchors
            mid = (start_min + end_min)//2
            lunch_target = clamp(mid, place_start + GAP//2, place_end - GAP//2)
            b1_target = clamp(lunch_target - GAP - 45, place_start, place_end)
            b3_target = clamp(lunch_target + GAP + 45, place_start, place_end)

            # lunch (30)
            lunch_c = win(lunch_target, BREAKS[1], place_start, place_end)
            lunch_s = choose_slot(lunch_c, BREAKS[1]); lunch_e = lunch_s + BREAKS[1]
            for t in time_list:
                if lunch_s <= t < lunch_e: break_load[t] += 1

            # first 15 (before lunch)
            b1_end_allowed = lunch_s - GAP
            b1_c = [t for t in win(b1_target, BREAKS[0], place_start, place_end) if (t+BREAKS[0]) <= b1_end_allowed]
            if not b1_c:
                b1_c = [t for t in time_list if place_start <= t <= max(place_start, lunch_s - GAP - BREAKS[0])]
            b1_s = choose_slot(b1_c, BREAKS[0]) if b1_c else None
            b1_e = b1_s + BREAKS[0] if b1_s is not None else None
            if b1_s is not None:
                for t in time_list:
                    if b1_s <= t < b1_e: break_load[t] += 1

            # last 15 (after lunch)
            b3_start_allowed = lunch_e + GAP
            b3_c = [t for t in win(b3_target, BREAKS[2], place_start, place_end) if t >= b3_start_allowed]
            if not b3_c:
                b3_c = [t for t in time_list if min(place_end - BREAKS[2], b3_target) <= t <= (place_end - BREAKS[2])]
            b3_s = choose_slot(b3_c, BREAKS[2]) if b3_c else None
            b3_e = b3_s + BREAKS[2] if b3_s is not None else None
            if b3_s is not None:
                for t in time_list:
                    if b3_s <= t < b3_e: break_load[t] += 1

            # attach
            item["breaks"] = []
            if b1_s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)), "end": minutes_to_hhmm(b1_e % (24*60)), "kind": f"break{BREAKS[0]}"})
            item["breaks"].append({"start": minutes_to_hhmm(lunch_s % (24*60)), "end": minutes_to_hhmm(lunch_e % (24*60)), "kind": f"lunch{BREAKS[1]}"})
            if b3_s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)), "end": minutes_to_hhmm(b3_e % (24*60)), "kind": f"break{BREAKS[2]}"})
    except Exception as e:
        print("[break planning] skipped due to error:", e)
        for item in roster:
            item.setdefault("breaks", [])

    summary = {
        "date": req.date,
        "intervals": [{"time": minutes_to_hhmm(t), "req": demand[t],
                       "assigned": assigned[t],
                       "on_break": 0,  # simple placeholder; next rev can subtract breaks from capacity
                       "working": assigned[t]} for t in times],
        "agents_used": len(used),
        "roster": roster,
        "notes": [
            "Greedy 9h assignment; fairness cap; site hours enforced (if enabled); 12h cross-day rest via prev_end_times.",
            "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes."
        ],
    }
    return summary

# ---------- Config (PATCH-like merge) ----------
@app.get("/config")
def get_config():
    return CONFIG

@app.put("/config")
def put_config(update: Dict[str, Any] = Body(...)):
    CONFIG.update(update or {})
    return CONFIG

# ---------- Simple HTML Playground ----------
@app.get("/playground", response_class=HTMLResponse)
def playground():
    return """<!doctype html><html><head><meta charset="utf-8"/>
<title>CCC Scheduler Playground</title>
<style>
  body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1024px;margin:24px auto;padding:0 16px;color:#333}
  input{padding:8px;font-size:14px;min-width:100px;max-width:240px}
  button{padding:8px 12px;margin-right:8px;background:#0b69ff;color:#fff;border:#0b69ff;border-radius:8px;cursor:pointer}
  .row{display:flex;gap:12px;flex-wrap:wrap}
  pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}
  label{display:block;margin:8px 0 4px}
</style>
</head>
<body>
<h2>CCC Scheduler — Playground</h2>
<p>Quickly try <code>/requirements</code> and <code>/generate-roster</code></p>
<div class="row">
  <div><label>Date</label><input id="date" value="2025-09-07"></div>
  <div><label>Language</label><input id="lang" value="EN"></div>
  <div><label>Group</label><input id="grp" value="G1"></div>
  <div><label>Shrinkage</label><input id="shr" value="0.30"></div>
  <div><label>Target SL</label><input id="tsl" value="0.8"></div>
  <div><label>Target T (sec)</label><input id="ttt" value="20"></div>
</div>
<p>
  <button onclick="runReq()">Run /requirements</button>
  <button onclick="runRoster()">Run /generate-roster</button>
  <button onclick="out.textContent=''">Clear output</button>
</p>
<pre id="out"></pre>
<script>
const out = document.getElementById('out');
function j(v){ out.textContent = JSON.stringify(v,null,2); }
function vals(){
  return {
    date: document.getElementById('date').value,
    language: document.getElementById('lang').value,
    grp: document.getElementById('grp').value,
    shrinkage: parseFloat(document.getElementById('shr').value),
    target_sl: parseFloat(document.getElementById('tsl').value),
    target_t: parseInt(document.getElementById('ttt').value,10),
  }
}
function runReq(){
  const v = vals();
  const q = new URLSearchParams({
    date: v.date, language: v.language, grp: v.grp,
    target_sl: v.target_sl, target_t: v.target_t
  });
  fetch('/requirements?'+q).then(r=>r.json()).then(j).catch(e=>j({error:String(e)}));
}
function runRoster(){
  const v = vals();
  fetch('/generate-roster',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(v)
  }).then(r=>r.json()).then(j).catch(e=>j({error:String(e)}));
}
</script>
</body></html>"""
