from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx, json

# -----------------------------------------------------------------------------
# Supabase REST setup
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

CONFIG_TABLE = "wfm_config"   # create if not exists (SQL shown below)
CONFIG_ID = "default"

# -----------------------------------------------------------------------------
# Defaults (used if no DB overrides exist)
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # staffing targets & intervals
    "interval_seconds": 1800,     # 30 min grid
    "target_sl": 0.80,            # 80% within target_t
    "target_t": 20,               # 20 sec
    "shrinkage": 0.30,            # 30%

    # shifts & breaks
    "shift_minutes": 9 * 60,      # 9 hours
    "break_pattern": [15, 30, 15],# 15 / 30 / 15
    "break_cap_frac": 0.25,       # at most 25% of assigned on break per interval
    "no_head": 60,                 # no breaks in first 60 minutes
    "no_tail": 60,                 # no breaks in last 60 minutes
    "lunch_gap": 120,             # min gap between lunch and each 15

    # rest & site windows (M4)
    "site_hours_enforced": False,  # set True to enforce
    "rest_min_minutes": 12 * 60,   # 12h cross-day rest
    "site_hours": {
        # Example:
        # "QA": {"open": "08:00", "close": "20:00"},
        # "IN": {"open": "08:00", "close": "22:00"},
        # "EU": {"open": "08:00", "close": "20:00"}
    },

    # misc
    "timezone": "UTC",
}

# simple in-process cache (soft TTL)
_CONFIG_CACHE: Dict[str, Any] = {}
_CONFIG_CACHE_TIME: Optional[datetime] = None
_CONFIG_TTL_SECONDS = 60  # refresh at most once per minute

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def get_config() -> Dict[str, Any]:
    global _CONFIG_CACHE, _CONFIG_CACHE_TIME
    now = datetime.utcnow()
    if _CONFIG_CACHE and _CONFIG_CACHE_TIME and (now - _CONFIG_CACHE_TIME).total_seconds() < _CONFIG_TTL_SECONDS:
        return _CONFIG_CACHE
    cfg = DEFAULT_CONFIG.copy()
    if not REST_BASE:
        _CONFIG_CACHE, _CONFIG_CACHE_TIME = cfg, now
        return cfg
    try:
        r = httpx.get(
            f"{REST_BASE}/{CONFIG_TABLE}",
            headers=HEADERS,
            params={"id": f"eq.{CONFIG_ID}", "select": "payload", "limit": 1},
            timeout=10,
        )
        if r.status_code == 200:
            rows = r.json()
            if rows:
                payload = rows[0].get("payload") or {}
                if isinstance(payload, dict):
                    cfg = _merge(DEFAULT_CONFIG, payload)
    except httpx.HTTPError:
        pass
    _CONFIG_CACHE, _CONFIG_CACHE_TIME = cfg, now
    return cfg

def set_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Upsert overrides into Supabase; returns merged config."""
    if not REST_BASE:
        merged = _merge(DEFAULT_CONFIG, overrides)
        global _CONFIG_CACHE, _CONFIG_CACHE_TIME
        _CONFIG_CACHE, _CONFIG_CACHE_TIME = merged, datetime.utcnow()
        return merged
    row = {"id": CONFIG_ID, "payload": overrides, "updated_at": datetime.utcnow().isoformat()}
    r = httpx.post(
        f"{REST_BASE}/{CONFIG_TABLE}",
        headers={**HEADERS, "Prefer": "resolution=merge-duplicates"},
        json=row,
        timeout=10,
    )
    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code, f"Config upsert failed: {r.text}")
    return get_config()

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="CCC Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later to your UI domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# -----------------------------------------------------------------------------
# Math helpers (requirements)
# -----------------------------------------------------------------------------
def erlang_c(a: float, N: int) -> float:
    if N <= a:  # stability guard
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
    lam = volume / max(1, interval_seconds)  # arrivals/sec
    mu = 1.0 / max(1, aht_sec)               # service rate per agent/sec
    a = lam / mu                             # offered load
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

# -----------------------------------------------------------------------------
# Public data endpoints
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Requirements (config-driven defaults)
# -----------------------------------------------------------------------------
@app.get("/requirements")
def requirements(
    date: str = Query(..., description="YYYY-MM-DD"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    target_sl: Optional[float] = None,
    target_t: Optional[int] = None,
    shrinkage: Optional[float] = None,
):
    cfg = get_config()
    interval_seconds = interval_seconds or cfg["interval_seconds"]
    target_sl       = target_sl       or cfg["target_sl"]
    target_t        = target_t        or cfg["target_t"]
    shrinkage       = shrinkage       if shrinkage is not None else cfg["shrinkage"]

    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date}",
        "order": "interval_time.asc,language.asc,grp.asc",
    }
    if language: params["language"] = f"eq.{language}"
    if grp:      params["grp"]      = f"eq.{grp}"

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
            "date": row["date"],
            "interval_time": row["interval_time"],
            "language": row["language"],
            "grp": row["grp"],
            "service": row["service"],
            "volume": vol,
            "aht_sec": aht,
            "req_core": N_core,
            "req_after_shrinkage": req_after,
        })
    return out

# -----------------------------------------------------------------------------
# Roster generation (config-driven)
# -----------------------------------------------------------------------------
class RosterRequest(BaseModel):
    date: str                    # YYYY-MM-DD
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None
    # M4: optional cross-day rest context: { "A001": "21:15", ... }
    prev_end_times: Optional[Dict[str, str]] = None

def time_to_minutes(tstr: str) -> int:
    hh, mm, _ss = tstr.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def parse_hhmm(s: str) -> Optional[int]:
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None

@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    cfg = get_config()

    # allow request-level overrides, else config defaults
    interval_seconds = req.interval_seconds or cfg["interval_seconds"]
    target_sl        = req.target_sl        or cfg["target_sl"]
    target_t         = req.target_t         or cfg["target_t"]
    shrinkage        = cfg["shrinkage"] if req.shrinkage is None else req.shrinkage

    SHIFT_MINUTES    = int(cfg["shift_minutes"])
    BREAK_CAP_FRAC   = float(cfg["break_cap_frac"])
    NO_HEAD          = int(cfg["no_head"])
    NO_TAIL          = int(cfg["no_tail"])
    LUNCH_GAP        = int(cfg["lunch_gap"])
    BP               = [int(x) for x in cfg["break_pattern"]]
    SITE_ENFORCE     = bool(cfg.get("site_hours_enforced", False))
    REST_MIN         = int(cfg.get("rest_min_minutes", 12*60))
    SITE_HOURS       = cfg.get("site_hours", {}) or {}

    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # 1) read forecasts for date/filter
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{req.date}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
    }, timeout=30)
    r.raise_for_status()
    fc_rows = r.json()

    # build per-interval requirement (after shrinkage)
    intervals = []
    for row in fc_rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = N_core if shrinkage >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        intervals.append({"t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after})

    # 2) fetch agents
    a = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params={
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": 2000
    }, timeout=30)
    a.raise_for_status()
    agents = a.json()

    # capability filter (naive)
    if req.language or req.grp:
        pool = [ag for ag in agents if all([
            (not req.language or (ag["primary_language"] == req.language or ag["secondary_language"] == req.language)),
            (not req.grp      or (req.grp in (ag.get("trained_groups") or [])))
        ])]
    else:
        pool = agents[:]

    # parse prev_end_times for rest rule (yesterday's end)
    prev_end_min: Dict[str, int] = {}
    if req.prev_end_times:
        for aid, hhmm in req.prev_end_times.items():
            m = parse_hhmm(hhmm)
            if m is not None:
                prev_end_min[aid] = m  # yesterday minute-of-day

    # pre-parse site hours into minutes for fast checks
    site_windows: Dict[str, Dict[str, int]] = {}
    for site, win in SITE_HOURS.items():
        o = parse_hhmm(win.get("open", "00:00"))
        c = parse_hhmm(win.get("close", "24:00"))
        if o is None: o = 0
        if c is None: c = 24*60
        site_windows[site] = {"open": o, "close": c}

    # 3) greedy shift placement to cover demand
    times = sorted({ time_to_minutes(x["t"]) for x in intervals })
    times_map = { t: [iv for iv in intervals if time_to_minutes(iv["t"]) == t] for t in times }
    demand = { t: sum(iv["req"] for iv in times_map[t]) for t in times }
    assigned = { t: 0 for t in times }

    roster = []
    used = set()

    def shift_gain(start_min):
        e = start_min + SHIFT_MINUTES
        g = 0
        for t in times:
            if start_min <= t < e:
                g += max(demand[t] - assigned[t], 0)
        return g

    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    def site_allows(agent, start_min, end_min):
        if not SITE_ENFORCE:
            return True
        site = agent.get("site_id")
        if not site or site not in site_windows:
            # if we don't know the site window, allow
            return True
        w = site_windows[site]
        # simple, no-overnight window; enforce start >= open and end <= close
        return (start_min >= w["open"]) and (end_min <= w["close"])

    def rest_allows(agent, start_min):
        aid = agent.get("agent_id")
        if aid in prev_end_min:
            # previous day end (minute-of-day). To compare to today's start:
            # gap = (start today) - (prev end yesterday) + 24h
            gap = (start_min - prev_end_min[aid]) + (24*60)
            return gap >= REST_MIN
        return True

    for start in candidate_starts:
        s_min = start
        e_min = s_min + SHIFT_MINUTES
        while any(s_min <= t < e_min and assigned[t] < demand[t] for t in times):
            pick = None
            for ag in pool:
                if ag["agent_id"] in used:
                    continue
                if not site_allows(ag, s_min, e_min):
                    continue
                if not rest_allows(ag, s_min):
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
                "shift": f"{minutes_to_hhmm(s_min)} - {minutes_to_hhmm(e_min % (24*60))}",
                "notes": "stub assignment"
            })
            for t in times:
                if s_min <= t < e_min:
                    assigned[t] += 1
            if all(assigned[t] >= demand[t] for t in times):
                break

    # 4) break planning with fairness cap & staggering (config-driven)
    def snap_to_grid(m):
        return min(times, key=lambda t: abs(t - m)) if times else m

    from math import ceil as _ceil
    break_load = {t: 0 for t in times}
    def cap_at(t):
        base = assigned[t]
        if base <= 0:
            return 0
        return max(1, int(_ceil(base * BREAK_CAP_FRAC)))

    def stress(s, dur):
        e = s + dur
        sc = 0
        for t in times:
            if s <= t < e:
                sc += max(demand[t] - assigned[t], 0)
        return sc

    def span_ok(s, dur):
        e = s + dur
        for t in times:
            if s <= t < e:
                if break_load[t] + 1 > cap_at(t):
                    return False
        return True

    def choose_slot(cands, dur):
        best = None; best_sc = None
        for s in cands:
            if span_ok(s, dur):
                sc = stress(s, dur)
                if best_sc is None or sc < best_sc:
                    best_sc = sc; best = s
        if best is not None:
            return best
        # fallback: minimal cap violation
        best = None; best_pen = None; best_sc = None
        for s in cands:
            e = s + dur
            pen = 0; sc = 0
            for t in times:
                if s <= t < e:
                    over = (break_load[t] + 1) - cap_at(t)
                    if over > 0: pen += over
                    sc += max(demand[t] - assigned[t], 0)
            if (best_pen is None or pen < best_pen or
               (pen == best_pen and (best_sc is None or sc < best_sc))):
                best_pen = pen; best_sc = sc; best = s
        return best

    def window_candidates(center, duration, lo, hi):
        w_lo = clamp(center - 60, lo, max(lo, hi - duration))
        w_hi = clamp(center + 60, lo, max(lo, hi - duration))
        cands = [t for t in times if w_lo <= t <= w_hi]
        return cands or [snap_to_grid(center)]

    b15a, blunch, b15b = BP[0], BP[1], BP[2]

    for item in roster:
        st_str, en_str = [s.strip() for s in item["shift"].split("-")]
        st_h, st_m = map(int, st_str.split(":"))
        en_h, en_m = map(int, en_str.split(":"))
        start_min = st_h*60 + st_m
        end_min = en_h*60 + en_m
        if end_min <= start_min:  # overnight guard
            end_min += 24*60

        place_start = start_min + NO_HEAD
        place_end   = end_min   - NO_TAIL
        if place_end - place_start < (b15a + blunch + b15b + 2*LUNCH_GAP):
            item["breaks"] = []
            continue

        mid = (start_min + end_min)//2
        lunch_target = clamp(mid, place_start + LUNCH_GAP//2, place_end - LUNCH_GAP//2)
        b1_target = clamp(lunch_target - LUNCH_GAP + (-15), place_start, place_end)
        b3_target = clamp(lunch_target + LUNCH_GAP +  15 , place_start, place_end)

        # Lunch first
        lunch_cands = window_candidates(lunch_target, blunch, place_start, place_end)
        lunch_s = choose_slot(lunch_cands, blunch) or snap_to_grid(lunch_target)
        lunch_e = lunch_s + blunch
        for t in times:
            if lunch_s <= t < lunch_e: break_load[t] += 1

        # First 15
        b1_end_allowed = lunch_s - LUNCH_GAP
        b1_cands = [t for t in window_candidates(b1_target, b15a, place_start, place_end)
                    if (t + b15a) <= b1_end_allowed]
        if not b1_cands:
            b1_cands = [t for t in times if place_start <= t <= max(place_start, lunch_s - LUNCH_GAP - b15a)]
        b1_s = choose_slot(b1_cands, b15a) if b1_cands else None
        if b1_s is None and place_start + b15a <= b1_end_allowed:
            b1_s = snap_to_grid(max(place_start, min(b1_target, b1_end_allowed - b15a)))
        b1_e = b1_s + b15a if b1_s is not None else None
        if b1_s is not None:
            for t in times:
                if b1_s <= t < b1_e: break_load[t] += 1

        # Last 15
        b3_start_allowed = lunch_e + LUNCH_GAP
        b3_cands = [t for t in window_candidates(b3_target, b15b, place_start, place_end)
                    if t >= b3_start_allowed]
        if not b3_cands:
            b3_cands = [t for t in times if min(place_end - b15b, b3_target) <= t <= (place_end - b15b)]
        b3_s = choose_slot(b3_cands, b15b) if b3_cands else None
        if b3_s is None and b3_start_allowed <= (place_end - b15b):
            b3_s = snap_to_grid(min(place_end - b15b, max(b3_target, b3_start_allowed)))
        b3_e = b3_s + b15b if b3_s is not None else None
        if b3_s is not None:
            for t in times:
                if b3_s <= t < b3_e: break_load[t] += 1

        # attach
        item["breaks"] = []
        if b1_s is not None:
            item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)),
                                   "end": minutes_to_hhmm(b1_e % (24*60)), "kind": f"break{b15a}"})
        item["breaks"].append({"start": minutes_to_hhmm(lunch_s % (24*60)),
                               "end": minutes_to_hhmm(lunch_e % (24*60)), "kind": f"lunch{blunch}"})
        if b3_s is not None:
            item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)),
                                   "end": minutes_to_hhmm(b3_e % (24*60)), "kind": f"break{b15b}"})

    # summary with current break load insight
    summary_intervals = []
    for t in times:
        onbreak = 0
        for it in roster:
            for b in it.get("breaks", []):
                # convert "HH:MM" back to minutes
                bs = parse_hhmm(b["start"])
                be = parse_hhmm(b["end"])
                if bs is not None and be is not None and bs <= t < be:
                    onbreak += 1
        summary_intervals.append({
            "time": minutes_to_hhmm(t),
            "req": demand[t],
            "assigned": assigned[t],
            "on_break": onbreak,
            "working": max(0, assigned[t] - onbreak),
        })

    return {
        "date": req.date,
        "intervals": summary_intervals,
        "agents_used": len(used),
        "roster": roster,
        "notes": [
            "Greedy 9h assignment; fairness cap; site hours enforced (if enabled); 12h cross-day rest via prev_end_times.",
            "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes."
        ],
    }

# -----------------------------------------------------------------------------
# Config endpoints
# -----------------------------------------------------------------------------
@app.get("/config")
def read_config():
    return get_config()

@app.put("/config")
def update_config(overrides: Dict[str, Any] = Body(...)):
    allowed = set(DEFAULT_CONFIG.keys())
    clean = {k: v for k, v in overrides.items() if k in allowed}
    if not clean:
        return get_config()
    new_cfg = set_config(_merge(DEFAULT_CONFIG, clean))
    return new_cfg

# -----------------------------------------------------------------------------
# Simple playground
# -----------------------------------------------------------------------------
PLAYGROUND_HTML = """
<!doctype html><html><head><meta charset="utf-8"/>
<title>CCC Scheduler Playground</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1024px;margin:24px auto;padding:0 16px}
label{display:block;margin:8px 0 4px;color:#333}
input{padding:8px;font-size:14px;width:100%;max-width:240px}
button{padding:8px 12px;margin-right:8px}
pre{background:#0b0f19;color:#d7e1ff;padding:12px;border-radius:8px;overflow:auto}
.row{display:flex;gap:12px;flex-wrap:wrap}
</style></head><body>
<h2>CCC Scheduler – Playground</h2>
<p>Quickly try <code>/requirements</code> and <code>/generate-roster</code></p>
<div class="row">
  <div><label>Date</label><input id="date" value="2025-09-07"/></div>
  <div><label>Language</label><input id="lang" value="EN"/></div>
  <div><label>Group</label><input id="grp" value="G1"/></div>
  <div><label>Shrinkage</label><input id="shr" value="0.30"/></div>
  <div><label>Target SL</label><input id="tsl" value="0.8"/></div>
  <div><label>Target T (sec)</label><input id="tt" value="20"/></div>
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
    target_t: parseInt(document.getElementById('tt').value,10),
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
  fetch('/generate-roster',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)})
  .then(r=>r.json()).then(j).catch(e=>j({error:String(e)}));
}
</script>
</body></html>
"""
@app.get("/playground")
def playground():
    return PLAYGROUND_HTML
