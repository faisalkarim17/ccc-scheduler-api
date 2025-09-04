# main.py — CCC Scheduler API (Supabase-backed, dd-mm-yyyy I/O)
# M10 explainability + free-text config  •  M11 shift-swaps (+ /rosters reader)

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx, json, logging, re
from fastapi.responses import StreamingResponse
from io import BytesIO

# --- RBAC helpers -------------------------------------------------------------
from functools import wraps
from fastapi import Request

ROLES = ("agent", "scheduler", "supervisor", "admin")

def _role_from_headers(request: Request) -> str:
    # Accept x-role from proxy/UI; default to "agent" if missing
    r = request.headers.get("x-role") or request.headers.get("X-Role") or ""
    r = r.strip().lower()
    return r if r in ROLES else "agent"

def _user_from_headers(request: Request) -> str:
    return (request.headers.get("x-user") or request.headers.get("X-User") or "user").strip()

def require_role(*allowed: str):
    def deco(fn):
        @wraps(fn)
        async def _async_wrapper(*args, **kwargs):
            request: Request = kwargs.get("request")
            if not request:
                # allow sync handlers that don't take request
                raise HTTPException(500, "RBAC middleware needs Request")
            role = _role_from_headers(request)
            if allowed and role not in allowed:
                raise HTTPException(403, f"Forbidden: requires roles {allowed}, but you are '{role}'")
            return await fn(*args, **kwargs)
        return _async_wrapper
    return deco


# ----------------------------------------------------------------------------- #
# Logging
# ----------------------------------------------------------------------------- #
log = logging.getLogger("ccc")
log.setLevel(logging.INFO)

# ----------------------------------------------------------------------------- #
# Supabase REST setup (READ via ANON; WRITE via SERVICE_ROLE if provided)
# ----------------------------------------------------------------------------- #
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # set in Railway

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS_READ = {}
    HEADERS_WRITE = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS_READ = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    # Prefer service key for writes
    _write_token = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    HEADERS_WRITE = {
        "apikey": _write_token,
        "Authorization": f"Bearer {_write_token}",
    }

# ----------------------------------------------------------------------------- #
# App + CORS
# ----------------------------------------------------------------------------- #
app = FastAPI(title="CCC Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------- #
# Helpers — dates & time (UI uses dd-mm-yyyy; DB uses yyyy-mm-dd)
# ----------------------------------------------------------------------------- #
def parse_ddmmyyyy(d: str) -> str:
    """Input dd-mm-yyyy -> output yyyy-mm-dd (ISO, for DB)."""
    try:
        return datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, f"Invalid date '{d}'. Use dd-mm-yyyy.")

def to_ddmmyyyy(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%m-%Y")

def hhmm_to_minutes(hhmm: str) -> int:
    """Accept 'HH:MM' or 'HH:MM:SS' and return minutes from midnight."""
    s = str(hhmm)
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(f"Bad time string '{hhmm}'")
    hh = int(parts[0]); mm = int(parts[1])
    return hh * 60 + mm

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_list(x) -> List[Any]:
    """Normalize Supabase text[]/json-ish fields to Python list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        # JSON array?
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
        # Fallback comma-split
        parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
        return [p for p in parts if p]
    return [x]

def is_weekend(iso_date: str) -> bool:
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    return dt.weekday() >= 5

def time_in_range(start_min: int, end_min: int, t: int) -> bool:
    if start_min <= end_min:
        return start_min <= t < end_min
    return t >= start_min or t < end_min

def get_daypart_bounds(conf_dayparts: Dict[str, List[str]]) -> Dict[str, Tuple[int, int]]:
    out = {}
    for k, pair in conf_dayparts.items():
        st = hhmm_to_minutes(pair[0]); en = hhmm_to_minutes(pair[1])
        out[k] = (st, en)
    return out

def infer_daypart(mins: int, bounds: Dict[str, Tuple[int,int]]) -> str:
    for name, (st, en) in bounds.items():
        if time_in_range(st, en, mins):
            return name
    if 6*60 <= mins < 14*60: return "morning"
    if 14*60 <= mins < 22*60: return "evening"
    return "night"

# ----------------------------------------------------------------------------- #
# CONFIG — default (merged from DB at startup)
# ----------------------------------------------------------------------------- #
CONFIG_DEFAULT: Dict[str, Any] = {
    # requirements
    "interval_seconds": 1800,
    "target_sl": 0.80,
    "target_t": 20,
    "shrinkage": 0.30,

    # scheduling
    "shift_minutes": 9 * 60,
    "break_pattern": [15, 30, 15],
    "break_cap_frac": 0.25,
    "no_head": 60,
    "no_tail": 60,
    "lunch_gap": 120,

    # constraints
    "site_hours_enforced": False,
    "site_hours": {},                 # {"QA":{"open":"10:00","close":"19:00"}}
    "rest_min_minutes": 12 * 60,
    "prev_end_times": {},

    "timezone": "UTC",

    # scoring
    "weight_primary_language": 2.0,
    "weight_secondary_language": 1.0,
    "weight_group_exact": 1.5,
    "weight_service_match": 1.0,

    # rules (buffers + fairness + hard caps)
    "rules": {
        "buffers": {
            "weekday_pct": 0.05,
            "weekend_pct": 0.10,
            "night_pct": 0.10
        },
        "dayparts": {
            "morning": ["06:00", "14:00"],
            "evening": ["14:00", "22:00"],
            "night":   ["22:00", "06:00"]
        },
        "hard": {
            "max_nights_per_7d": 2,
            # New (only enforced if not None):
            "swap_min_rest_minutes": 12 * 60,   # fallbacks to rest_min_minutes if None
            "swap_max_days_per_7d": None,       # e.g. 6
            "swap_min_hours_per_7d": None,      # e.g. 36
            "swap_max_hours_per_7d": None       # e.g. 60
        },

        "soft": {
            "balance_dayparts": True,
            "weight_balance": 1.0
        }
    },

    # fairness ledger
    "fairness_ledger": {}
}
CONFIG: Dict[str, Any] = dict(CONFIG_DEFAULT)

async def load_config_from_db():
    if not REST_BASE:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{REST_BASE}/wfm_config",
                headers=HEADERS_READ,
                params={"select": "id,config", "id": "eq.1", "limit": 1},
            )
            if r.status_code == 200 and r.json():
                db_conf = r.json()[0].get("config") or {}
                CONFIG.update({k: db_conf[k] for k in db_conf})
    except Exception:
        pass

async def save_config_to_db():
    if not REST_BASE:
        return
    payload = {"id": 1, "config": CONFIG}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{REST_BASE}/wfm_config",
            headers={**HEADERS_WRITE, "Content-Type": "application/json"},
            params={"on_conflict": "id"},
            content=json.dumps([payload]),
        )
        if r.status_code not in (200, 201):
            raise HTTPException(r.status_code, r.text)

@app.on_event("startup")
async def _startup():
    await load_config_from_db()

# ----------------------------------------------------------------------------- #
# Health
# ----------------------------------------------------------------------------- #
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
            headers=HEADERS_READ,
            params={"select": "agent_id", "limit": 1},
            timeout=10
        )
        r.raise_for_status()
        return {"db": "ok", "agents_table": "reachable"}
    except httpx.HTTPError as e:
        raise HTTPException(500, f"Supabase REST error: {e}")

# ----------------------------------------------------------------------------- #
# Erlang C & requirements
# ----------------------------------------------------------------------------- #
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

# ----------------------------------------------------------------------------- #
# DB helpers
# ----------------------------------------------------------------------------- #
def sb_get(table: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    r = httpx.get(f"{REST_BASE}/{table}", headers=HEADERS_READ, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()

def sb_post(table: str, rows: List[Dict[str, Any]]) -> None:
    r = httpx.post(
        f"{REST_BASE}/{table}",
        headers={**HEADERS_WRITE, "Content-Type": "application/json"},
        content=json.dumps(rows),
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code, r.text)

def sb_upsert(table: str, rows: List[Dict[str, Any]], on_conflict: str) -> None:
    r = httpx.post(
        f"{REST_BASE}/{table}",
        headers={**HEADERS_WRITE, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
        params={"on_conflict": on_conflict},
        content=json.dumps(rows),
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code, r.text)

# ----------------------------------------------------------------------------- #
# Thin list endpoints
# ----------------------------------------------------------------------------- #
@app.get("/agents")
def list_agents(
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    site_id: Optional[str] = None,
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    sel = "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services"
    params = {"select": sel, "limit": limit, "offset": offset, "order": "agent_id.asc"}
    if site_id:
        params["site_id"] = f"eq.{site_id}"
    if language:
        params["or"] = f"(primary_language.eq.{language},secondary_language.eq.{language})"
    if grp:
        params["trained_groups"] = f"cs.{{{grp}}}"
    rows = sb_get("agents", params)
    for ag in rows:
        ag["trained_groups"] = ensure_list(ag.get("trained_groups"))
        ag["trained_services"] = ensure_list(ag.get("trained_services"))
    return rows

@app.get("/forecasts")
def list_forecasts(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    iso = parse_ddmmyyyy(date)
    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
        "order": "interval_time.asc,language.asc,grp.asc,service.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"
    if service:
        params["service"] = f"eq.{service}"
    rows = sb_get("forecasts", params)
    for row in rows:
        row["date"] = to_ddmmyyyy(row["date"])
    return rows

# ----------------------------------------------------------------------------- #
# Requirements (per-interval) with buffers
# ----------------------------------------------------------------------------- #
def _buffer_multiplier(iso_date: str, interval_hhmm: str) -> float:
    rules = CONFIG.get("rules", {})
    buffers = rules.get("buffers", {})
    dayparts = rules.get("dayparts", {})
    bounds = get_daypart_bounds(dayparts) if dayparts else {}
    base = 1.0
    base += buffers.get("weekend_pct", 0.0) if is_weekend(iso_date) else buffers.get("weekday_pct", 0.0)
    if bounds:
        iv_min = hhmm_to_minutes(interval_hhmm)
        dp = infer_daypart(iv_min, bounds)
        if dp == "night":
            base += buffers.get("night_pct", 0.0)
    return base

@app.get("/requirements")
def requirements(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
    interval_seconds: Optional[int] = None,
    target_sl: Optional[float] = None,
    target_t: Optional[int] = None,
    shrinkage: Optional[float] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    interval_seconds = interval_seconds or CONFIG["interval_seconds"]
    target_sl = CONFIG["target_sl"] if target_sl is None else target_sl
    target_t = CONFIG["target_t"] if target_t is None else target_t
    shrinkage = CONFIG["shrinkage"] if shrinkage is None else shrinkage

    iso = parse_ddmmyyyy(date)
    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
        "order": "interval_time.asc,language.asc,grp.asc,service.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"
    if service:
        params["service"] = f"eq.{service}"

    rows = sb_get("forecasts", params)
    out: List[Dict[str, Any]] = []
    for row in rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = N_core if shrinkage >= 0.999 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        mult = _buffer_multiplier(iso, str(row["interval_time"]))
        req_buf = int(ceil(req_after * mult))
        out.append({
            "date": to_ddmmyyyy(row["date"]),
            "interval_time": row["interval_time"],
            "language": row["language"],
            "grp": row["grp"],
            "service": row["service"],
            "volume": vol,
            "aht_sec": aht,
            "req_core": N_core,
            "req_after_shrinkage": req_after,
            "req_with_buffers": req_buf
        })
    return out

# ----------------------------------------------------------------------------- #
# Models
# ----------------------------------------------------------------------------- #
class RosterRequest(BaseModel):
    date: str
    language: Optional[str] = None
    grp: Optional[str] = None
    service: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None

class RangeRequest(BaseModel):
    date_from: str
    date_to: str
    language: Optional[str] = None
    grp: Optional[str] = None
    service: Optional[str] = None
    persist: Optional[bool] = False

class SaveRosterRequest(BaseModel):
    date: str
    roster: List[Dict[str, Any]]

class ExplainRequest(BaseModel):
    date: str
    language: Optional[str] = None
    grp: Optional[str] = None
    service: Optional[str] = None
    agent_id: Optional[str] = None

class ParseConfigRequest(BaseModel):
    text: str

# M11 swap models
class SwapRequestIn(BaseModel):
    date: str                # dd-mm-yyyy
    agent_from: str
    agent_to: str
    reason: Optional[str] = None
    force: Optional[bool] = False  # auto-approve & apply immediately

class SwapDecisionIn(BaseModel):
    id: str                  # UUID of swap request
    approver: Optional[str] = None
    notes: Optional[str] = None
    
class SwapRespondIn(BaseModel):
    id: str
    actor_agent_id: str          # the agent who is responding (should be agent_to)
    decision: str                # "accept" or "decline"
    notes: Optional[str] = None

# ----------------------------------------------------------------------------- #
# Fairness helpers (last 7 days)
# ----------------------------------------------------------------------------- #
def _ledger_prune_older_than(days: int = 30):
    led = CONFIG.get("fairness_ledger", {})
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    for aid, buckets in list(led.items()):
        for dp, arr in list(buckets.items()):
            if isinstance(arr, list):
                led[aid][dp] = [d for d in arr if d >= cutoff]

def _ledger_count_in_window(aid: str, daypart: str, iso_end: str, window_days: int = 7) -> int:
    led = CONFIG.get("fairness_ledger", {})
    arr = (led.get(aid, {}).get(daypart) or [])
    if not arr:
        return 0
    end = datetime.strptime(iso_end, "%Y-%m-%d")
    start = end - timedelta(days=window_days-1)
    s = start.strftime("%Y-%m-%d"); e = end.strftime("%Y-%m-%d")
    return sum(1 for d in arr if s <= d <= e)

def _ledger_add(aid: str, daypart: str, iso_date: str):
    if "fairness_ledger" not in CONFIG:
        CONFIG["fairness_ledger"] = {}
    if aid not in CONFIG["fairness_ledger"]:
        CONFIG["fairness_ledger"][aid] = {"morning": [], "evening": [], "night": []}
    if iso_date not in CONFIG["fairness_ledger"][aid][daypart]:
        CONFIG["fairness_ledger"][aid][daypart].append(iso_date)

# ----------------------------------------------------------------------------- #
# Scoring — skills + service + fairness
# ----------------------------------------------------------------------------- #
def agent_score(agent: Dict[str, Any],
                language: Optional[str],
                grp: Optional[str],
                service: Optional[str],
                target_daypart: Optional[str],
                iso_date: str) -> Tuple[float, Dict[str, float]]:
    score = 0.0
    breakdown = {}
    if language and agent.get("primary_language") == language:
        w = CONFIG.get("weight_primary_language", 2.0); score += w; breakdown["primary_language"] = w
    if language and agent.get("secondary_language") == language:
        w = CONFIG.get("weight_secondary_language", 1.0); score += w; breakdown["secondary_language"] = w
    if grp and (grp in ensure_list(agent.get("trained_groups"))):
        w = CONFIG.get("weight_group_exact", 1.5); score += w; breakdown["group_match"] = w
    if service and (service in ensure_list(agent.get("trained_services"))):
        w = CONFIG.get("weight_service_match", 1.0); score += w; breakdown["service_match"] = w

    rules = CONFIG.get("rules", {})
    soft = rules.get("soft", {})
    if soft.get("balance_dayparts") and target_daypart:
        w = float(soft.get("weight_balance", 1.0))
        had = _ledger_count_in_window(agent["agent_id"], target_daypart, iso_date, window_days=7)
        adj = 0.0
        if had == 0: adj = +1.0 * w
        elif had == 1: adj = +0.5 * w
        elif had >= 3: adj = -1.0 * w
        score += adj
        breakdown["fairness_balance"] = adj
    return score, breakdown

# ----------------------------------------------------------------------------- #
# Generate roster (single day) — M9 logic (unchanged)
# ----------------------------------------------------------------------------- #
def _generate_intervals(iso: str, req: RosterRequest,
                        interval_seconds: int, target_sl: float, target_t: int, shrinkage: float):
    f_params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
        "order": "interval_time.asc,language.asc,grp.asc,service.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
        **({"service": f"eq.{req.service}"} if req.service else {}),
    }
    fc_rows = sb_get("forecasts", f_params)
    intervals = []
    for row in fc_rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = N_core if shrinkage >= 0.999 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        mult = _buffer_multiplier(iso, str(row["interval_time"]))
        req_buf = int(ceil(req_after * mult))
        intervals.append({
            "t": row["interval_time"], "lang": row["language"],
            "grp": row["grp"], "svc": row["service"], "req": req_buf
        })
    return intervals

@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    try:
        if not REST_BASE:
            raise HTTPException(500, "Supabase env vars missing")
        interval_seconds = req.interval_seconds or CONFIG["interval_seconds"]
        target_sl = req.target_sl if req.target_sl is not None else CONFIG["target_sl"]
        target_t = req.target_t if req.target_t is not None else CONFIG["target_t"]
        shrinkage = req.shrinkage if req.shrinkage is not None else CONFIG["shrinkage"]
        iso = parse_ddmmyyyy(req.date)
        rules = CONFIG.get("rules", {})
        dayparts_conf = rules.get("dayparts", {})
        dp_bounds = get_daypart_bounds(dayparts_conf) if dayparts_conf else {}
        intervals = _generate_intervals(iso, req, interval_seconds, target_sl, target_t, shrinkage)

        sel = "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services"
        agents = sb_get("agents", {"select": sel, "limit": 2000})
        for ag in agents:
            ag["trained_groups"] = ensure_list(ag.get("trained_groups"))
            ag["trained_services"] = ensure_list(ag.get("trained_services"))

        if req.language or req.grp or req.service:
            pool = [
                ag for ag in agents
                if (not req.language or (ag.get("primary_language") == req.language or ag.get("secondary_language") == req.language))
                and (not req.grp or (req.grp in ag.get("trained_groups", [])))
                and (not req.service or (req.service in ag.get("trained_services", []) or not ag.get("trained_services")))
            ]
        else:
            pool = agents[:]

        times = sorted({hhmm_to_minutes(x["t"]) for x in intervals})
        times_map: Dict[int, List[Dict[str, Any]]] = {t: [iv for iv in intervals if hhmm_to_minutes(iv["t"]) == t] for t in times}
        demand = {t: sum(iv["req"] for iv in times_map[t]) for t in times}
        assigned = {t: 0 for t in times}

        SHIFT_MIN = CONFIG["shift_minutes"]
        roster: List[Dict[str, Any]] = []
        used = set()

        def shift_gain(start_min):
            end_min = start_min + SHIFT_MIN
            return sum(max(demand[t] - assigned[t], 0) for t in times if start_min <= t < end_min)

        candidate_starts = sorted({t for t in times}, key=lambda s: -shift_gain(s))

        hard = rules.get("hard", {})
        max_nights_7d = int(hard.get("max_nights_per_7d", 2))

        def start_dp(start_min: int) -> str:
            return infer_daypart(start_min, dp_bounds or get_daypart_bounds(CONFIG["rules"]["dayparts"]))

        for start in candidate_starts:
            start_min = start
            end_min = start_min + SHIFT_MIN
            target_dp = start_dp(start_min)
            while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
                ranked = sorted(
                    [ag for ag in pool if ag["agent_id"] not in used],
                    key=lambda ag: agent_score(ag, req.language, req.grp, req.service, target_dp, iso)[0],
                    reverse=True
                )
                pick = None
                for ag in ranked:
                    if CONFIG["site_hours_enforced"]:
                        site = ag.get("site_id"); sh = CONFIG["site_hours"].get(site) if site else None
                        if sh and sh.get("open") and sh.get("close"):
                            o = hhmm_to_minutes(sh["open"]); c = hhmm_to_minutes(sh["close"])
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
                    if target_dp == "night":
                        night_ct = _ledger_count_in_window(ag["agent_id"], "night", iso, window_days=7)
                        if night_ct >= max_nights_7d:
                            continue
                    pick = ag; break
                if not pick:
                    break
                used.add(pick["agent_id"])
                roster.append({
                    "agent_id": pick["agent_id"],
                    "full_name": pick["full_name"],
                    "site_id": pick.get("site_id"),
                    "date": req.date,
                    "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                    "service": req.service or "",
                    "daypart": target_dp,
                    "notes": "auto assignment",
                })
                for t in times:
                    if start_min <= t < end_min:
                        assigned[t] += 1
                if all(assigned[t] >= demand[t] for t in times):
                    break

        # Break planning (unchanged)
        try:
            time_list = sorted(times)
            if time_list:
                def snap_to_grid(m): return min(time_list, key=lambda t: abs(t - m))
                BREAK_CAP_FRAC = CONFIG["break_cap_frac"]
                break_load = {t: 0 for t in time_list}
                def cap_at(t):
                    base = assigned[t]
                    if base <= 0: return 0
                    return max(1, int(ceil(base * BREAK_CAP_FRAC)))
                def span_ok(s, dur):
                    e = s + dur
                    for t in time_list:
                        if s <= t < e and (break_load[t] + 1 > cap_at(t)):
                            return False
                    return True
                def stress(s, dur):
                    e = s + dur; sc = 0
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
                    if best is not None: return best
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
                    start_min = st_h * 60 + st_m
                    end_min   = en_h * 60 + en_m
                    if end_min <= start_min: end_min += 24 * 60
                    NO_HEAD, NO_TAIL, GAP = CONFIG["no_head"], CONFIG["no_tail"], CONFIG["lunch_gap"]
                    place_start = start_min + NO_HEAD
                    place_end   = end_min   - NO_TAIL
                    pattern = CONFIG["break_pattern"]
                    if place_end - place_start < (sum(pattern) + 2 * GAP):
                        item["breaks"] = []; continue
                    mid = (start_min + end_min) // 2
                    lunch_target = clamp(mid, place_start + GAP // 2, place_end - GAP // 2)
                    lunch_dur = max(10, int(pattern[1]))
                    lunch_cands = window_candidates(lunch_target, lunch_dur, place_start, place_end)
                    lunch_s = choose_slot(lunch_cands, lunch_dur) or lunch_cands[0]
                    lunch_e = lunch_s + lunch_dur
                    for t in time_list:
                        if lunch_s <= t < lunch_e: break_load[t] += 1
                    b1_dur = max(5, int(pattern[0])); b1_end_allowed = lunch_s - GAP
                    b1_cands = [t for t in window_candidates(lunch_s - GAP - 45, b1_dur, place_start, place_end)
                                if (t + b1_dur) <= b1_end_allowed]
                    b1_s = b1_cands and choose_slot(b1_cands, b1_dur)
                    b1_e = (b1_s + b1_dur) if b1_s else None
                    if b1_s:
                        for t in time_list:
                            if b1_s <= t < b1_e: break_load[t] += 1
                    b3_dur = max(5, int(pattern[2])); b3_start_allowed = lunch_e + GAP
                    b3_cands = [t for t in window_candidates(lunch_s + GAP + 45, b3_dur, place_start, place_end)
                                if t >= b3_start_allowed]
                    b3_s = b3_cands and choose_slot(b3_cands, b3_dur)
                    b3_e = (b3_s + b3_dur) if b3_s else None
                    if b3_s:
                        for t in time_list:
                            if b3_s <= t < b3_e: break_load[t] += 1
                    item["breaks"] = []
                    if b1_s:
                        item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)),
                                               "end":   minutes_to_hhmm(b1_e % (24*60)), "kind": f"break{b1_dur}"})
                    item["breaks"].append({"start": minutes_to_hhmm(lunch_s % (24*60)),
                                           "end":   minutes_to_hhmm(lunch_e % (24*60)), "kind": f"lunch{lunch_dur}"})
                    if b3_s:
                        item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)),
                                               "end":   minutes_to_hhmm(b3_e % (24*60)), "kind": f"break{b3_dur}"})
        except Exception as e:
            log.info(f"[break planning] skipped: {e}")
            for item in roster: item.setdefault("breaks", [])

        summary = {
            "date": req.date,
            "intervals": [{"time": minutes_to_hhmm(t), "req": demand.get(t, 0),
                           "assigned": assigned.get(t, 0),
                           "on_break": 0, "working": assigned.get(t, 0)} for t in times],
            "agents_used": len(used),
            "roster": roster,
            "notes": [
                "Greedy assignment; fairness soft-rule; night cap hard-rule; buffers applied (weekday/weekend/night).",
                "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes, rules, fairness_ledger.",
            ],
        }
        return summary
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[generate_roster] ERROR: {e}")
        raise HTTPException(500, "Internal Server Error")

# ----------------------------------------------------------------------------- #
# Generate roster for a range (+ optional persist) — ledger + prev_end
# ----------------------------------------------------------------------------- #
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
            grp=req.grp,
            service=req.service,
        )
        day = generate_roster(day_req)
        out[cur.strftime("%Y-%m-%d")] = day

        if req.persist and day.get("roster"):
            iso = cur.strftime("%Y-%m-%d")
            rows = []
            for item in day["roster"]:
                dp = item.get("daypart") or "morning"
                _ledger_add(item["agent_id"], dp, iso)
                rows.append({
                    "date": iso,
                    "agent_id": item.get("agent_id"),
                    "full_name": item.get("full_name"),
                    "site_id": item.get("site_id"),
                    "shift": item.get("shift"),
                    "breaks": item.get("breaks") or [],
                    "meta": item,
                })
                try:
                    sh = item.get("shift","")
                    st_str, en_str = [s.strip() for s in sh.split("-")]
                    en_h, en_m = map(int, en_str.split(":"))
                    end_dt = datetime.strptime(iso + f"T{en_h:02d}:{en_m:02d}:00", "%Y-%m-%dT%H:%M:%S")
                    st_h, st_m = map(int, st_str.split(":"))
                    st_dt = datetime.strptime(iso + f"T{st_h:02d}:{st_m:02d}:00", "%Y-%m-%dT%H:%M:%S")
                    if end_dt <= st_dt: end_dt = end_dt + timedelta(days=1)
                    CONFIG["prev_end_times"][item["agent_id"]] = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception as e:
                    log.info(f"[prev_end_times update skipped] {e}")
            if rows:
                sb_post("rosters", rows)
        cur += timedelta(days=1)

    if req.persist:
        _ledger_prune_older_than(45)
        try:
            import anyio
            anyio.from_thread.run(save_config_to_db)
        except Exception:
            pass

    return out

# ----------------------------------------------------------------------------- #
# Save roster (single day) to Supabase — also ledger update
# ----------------------------------------------------------------------------- #
@app.post("/save-roster")
def save_roster(req: SaveRosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    if not isinstance(req.roster, list):
        raise HTTPException(400, "roster must be a list")
    iso = parse_ddmmyyyy(req.date)
    rows = []
    for item in req.roster:
        dp = (item.get("daypart") or "morning")
        _ledger_add(item.get("agent_id",""), dp, iso)
        rows.append({
            "date": iso,
            "agent_id": item.get("agent_id"),
            "full_name": item.get("full_name"),
            "site_id": item.get("site_id"),
            "shift": item.get("shift"),
            "breaks": item.get("breaks") or [],
            "meta": item,
        })
    if not rows:
        return {"status": "ok", "inserted": 0}
    sb_post("rosters", rows)
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return {"status": "ok", "inserted": len(rows)}

# ----------------------------------------------------------------------------- #
# Runtime config endpoints
# ----------------------------------------------------------------------------- #
@app.get("/config")
def get_config():
    return CONFIG

@app.put("/config")
def put_config(body: Dict[str, Any] = Body(...)):
    for k, v in body.items():
        if k in CONFIG_DEFAULT or k in ("rules", "fairness_ledger", "prev_end_times"):
            CONFIG[k] = v
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return CONFIG

@app.get("/admin/config/export", response_class=PlainTextResponse)
def export_config():
    return json.dumps(CONFIG, indent=2, sort_keys=True)

@app.post("/admin/config/import")
def import_config(new_conf: Dict[str, Any] = Body(...)):
    allowed = set(CONFIG_DEFAULT.keys()) | {"rules","fairness_ledger","prev_end_times"}
    applied = {k: new_conf[k] for k in new_conf if k in allowed}
    CONFIG.update(applied)
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return {"status": "ok", "applied_keys": sorted(applied.keys())}

@app.post("/admin/config/reset-prev-end")
def reset_prev_end():
    n = len(CONFIG.get("prev_end_times", {}))
    CONFIG["prev_end_times"] = {}
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return {"status": "ok", "cleared": n}

@app.post("/admin/config/reset-ledger")
def reset_ledger():
    n = len(CONFIG.get("fairness_ledger", {}))
    CONFIG["fairness_ledger"] = {}
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return {"status":"ok","cleared_agents": n}

# --- Config versioning (M17) --------------------------------------------------
class ConfigVersionIn(BaseModel):
    scope_level: Optional[str] = "global"  # global | site | service
    scope_id: Optional[str] = None         # e.g., "QA" for site
    notes: Optional[str] = None

@app.get("/config/version/list")
@require_role("scheduler", "supervisor", "admin")
async def config_version_list(request: Request, limit: int = 50):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    rows = sb_get("wfm_config_versions", {
        "select": "id,created_at,author,scope_level,scope_id,notes",
        "order": "created_at.desc",
        "limit": limit
    })
    return rows

@app.post("/config/version/save")
@require_role("scheduler", "supervisor", "admin")
async def config_version_save(body: ConfigVersionIn, request: Request):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    author = _user_from_headers(request)
    row = {
        "author": author,
        "scope_level": (body.scope_level or "global"),
        "scope_id": body.scope_id,
        "notes": body.notes or "",
        "config": CONFIG
    }
    sb_post("wfm_config_versions", [row])
    return {"status":"ok","saved":True}

class ConfigRollbackIn(BaseModel):
    id: str

@app.post("/config/version/rollback")
@require_role("admin")
async def config_version_rollback(body: ConfigRollbackIn, request: Request):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    rows = sb_get("wfm_config_versions", {"select": "id,config", "id": f"eq.{body.id}", "limit": 1})
    if not rows:
        raise HTTPException(404, "version id not found")
    version = rows[0]["config"] or {}
    # Apply entirely
    for k, v in version.items():
        CONFIG[k] = v
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return {"status":"ok","rolled_back_to": body.id}
# --- What-if simulate ---------------------------------------------------------
class SimulateIn(BaseModel):
    policy_text: str
    date_from: Optional[str] = None  # dd-mm-yyyy
    date_to: Optional[str] = None    # dd-mm-yyyy
    language: Optional[str] = None
    grp: Optional[str] = None
    service: Optional[str] = None

@app.post("/simulate")
@require_role("scheduler", "supervisor", "admin")
async def simulate(body: SimulateIn, request: Request):
    # 1) parse policy to proposed CONFIG deltas (no apply)
    parsed = parse_config(ParseConfigRequest(text=body.policy_text), apply=False)
    proposed = parsed.get("proposed_changes") or {}

    # 2) If dates provided, recompute requirements with a temporary overlay
    def overlay(cfg: Dict[str, Any]):
        snap = dict(CONFIG)  # shallow is fine (sub-dicts are read-only in this flow)
        # merge rules carefully
        for k, v in proposed.items():
            if k == "rules":
                cur = snap.setdefault("rules", {})
                for rk, rv in v.items():
                    if isinstance(rv, dict):
                        cur.setdefault(rk, {})
                        cur[rk].update(rv)
                    else:
                        cur[rk] = rv
            else:
                snap[k] = v
        return snap

    impact = None
    if body.date_from and body.date_to:
        iso_from = parse_ddmmyyyy(body.date_from); iso_to = parse_ddmmyyyy(body.date_to)
        # quick roll-up: per day total req_with_buffers
        cur = datetime.strptime(iso_from, "%Y-%m-%d")
        end = datetime.strptime(iso_to, "%Y-%m-%d")
        roll = []
        while cur <= end:
            date_dd = to_ddmmyyyy(cur.strftime("%Y-%m-%d"))
            # temporarily run requirements using overlay
            snap = overlay(CONFIG)
            # monkey-patch globals just for this call
            saved_rules = CONFIG.get("rules")
            saved_target_sl = CONFIG.get("target_sl"); saved_target_t = CONFIG.get("target_t")
            saved_shrinkage = CONFIG.get("shrinkage"); saved_interval = CONFIG.get("interval_seconds")
            try:
                CONFIG["rules"] = snap.get("rules", CONFIG.get("rules"))
                CONFIG["target_sl"] = snap.get("target_sl", CONFIG.get("target_sl"))
                CONFIG["target_t"] = snap.get("target_t", CONFIG.get("target_t"))
                CONFIG["shrinkage"] = snap.get("shrinkage", CONFIG.get("shrinkage"))
                CONFIG["interval_seconds"] = snap.get("interval_seconds", CONFIG.get("interval_seconds"))
                reqs = requirements(date=date_dd, language=body.language, grp=body.grp, service=body.service)
                total = sum(r["req_with_buffers"] for r in reqs)
                roll.append({"date": date_dd, "total_req": total})
            finally:
                CONFIG["rules"] = saved_rules
                CONFIG["target_sl"] = saved_target_sl; CONFIG["target_t"] = saved_target_t
                CONFIG["shrinkage"] = saved_shrinkage; CONFIG["interval_seconds"] = saved_interval
            cur += timedelta(days=1)
        impact = {"req_totals": roll}

    return {"proposed_changes": proposed, "impact": impact}

# --- AI proxy (provider=ollama|openai) ---------------------------------------
class AIIn(BaseModel):
    prompt: str
    provider: Optional[str] = "ollama"
    model: Optional[str] = None  # defaults per provider

@app.post("/ai/complete")
def ai_complete(body: AIIn):
    prov = (body.provider or "ollama").lower()
    text = body.prompt or ""
    if not text.strip():
        raise HTTPException(400, "prompt is required")

    # ENV config
    OLLAMA_HOST = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    try:
        if prov == "ollama":
            model = body.model or "llama3.1:8b-instruct-fp16"
            url = f"{OLLAMA_HOST}/api/generate"
            r = httpx.post(url, json={"model": model, "prompt": text, "stream": False}, timeout=60)
            if r.status_code != 200:
                return {"provider":"ollama","ok":False,"error":r.text}
            out = r.json()
            return {"provider":"ollama","ok":True,"model":model,"completion": out.get("response","")}
        elif prov == "openai":
            if not OPENAI_KEY:
                return {"provider":"openai","ok":False,"error":"OPENAI_API_KEY not configured"}
            model = body.model or "gpt-4o-mini"
            # lightweight REST call (no SDK)
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={"model": model, "messages":[{"role":"user","content": text}], "temperature": 0.2},
                timeout=60
            )
            if r.status_code != 200:
                return {"provider":"openai","ok":False,"error":r.text}
            data = r.json()
            msg = (data.get("choices") or [{}])[0].get("message",{}).get("content","")
            return {"provider":"openai","ok":True,"model":model,"completion": msg}
        else:
            return {"ok":False,"error":"unknown provider"}
    except Exception as e:
        return {"ok":False,"error": str(e)}

@app.get("/swaps/audit")
@require_role("supervisor", "admin")
async def swaps_audit(date_from: Optional[str] = None, date_to: Optional[str] = None, limit: int = 200, request: Request = None):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    params = {"select":"*","order":"decision_at.desc","limit": limit}
    if date_from:
        params["decision_at"] = f"gte.{parse_ddmmyyyy(date_from)}"
    rows = sb_get("swaps_audit", params)
    if date_to:
        iso_to = parse_ddmmyyyy(date_to)
        rows = [r for r in rows if (r.get("decision_at") or "")[:10] <= iso_to]
    return rows


# ----------------------------------------------------------------------------- #
# NEW — M10: Explainability
# ----------------------------------------------------------------------------- #
@app.post("/explain-assignment")
def explain_assignment(req: ExplainRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    r = RosterRequest(
        date=req.date, language=req.language, grp=req.grp, service=req.service,
        interval_seconds=CONFIG["interval_seconds"], target_sl=CONFIG["target_sl"],
        target_t=CONFIG["target_t"], shrinkage=CONFIG["shrinkage"]
    )
    iso = parse_ddmmyyyy(r.date)
    rules = CONFIG.get("rules", {})
    dp_bounds = get_daypart_bounds(rules.get("dayparts", {}))
    intervals = _generate_intervals(iso, r, r.interval_seconds, r.target_sl, r.target_t, r.shrinkage)
    if not intervals:
        return {"explain": "No forecast intervals on this date with given filters.", "eligible": False}

    times = sorted({hhmm_to_minutes(x["t"]) for x in intervals})
    SHIFT_MIN = CONFIG["shift_minutes"]
    def shift_gain(start_min):
        demand = {t: sum(iv["req"] for iv in intervals if hhmm_to_minutes(iv["t"])==t) for t in times}
        end_min = start_min + SHIFT_MIN
        return sum(max(demand[t], 0) for t in times if start_min <= t < end_min)
    candidate_starts = sorted({t for t in times}, key=lambda s: -shift_gain(s))
    target_dp = infer_daypart(candidate_starts[0], dp_bounds or get_daypart_bounds(CONFIG["rules"]["dayparts"]))

    sel = "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services"
    agents = sb_get("agents", {"select": sel, "limit": 2000})
    for ag in agents:
        ag["trained_groups"] = ensure_list(ag.get("trained_groups"))
        ag["trained_services"] = ensure_list(ag.get("trained_services"))

    def eligible(ag):
        if r.language and not (ag.get("primary_language")==r.language or ag.get("secondary_language")==r.language):
            return False, "language_mismatch"
        if r.grp and (r.grp not in ag.get("trained_groups", [])):
            return False, "group_not_trained"
        if r.service and (r.service not in ag.get("trained_services", []) and ag.get("trained_services")):
            return False, "service_not_trained"
        if CONFIG["site_hours_enforced"]:
            site = ag.get("site_id"); sh = CONFIG["site_hours"].get(site) if site else None
            if sh and sh.get("open") and sh.get("close"):
                o = hhmm_to_minutes(sh["open"]); c = hhmm_to_minutes(sh["close"])
                st = candidate_starts[0]; en = st + SHIFT_MIN
                if not (o <= st and en <= c):
                    return False, "outside_site_hours"
        prev_end = CONFIG["prev_end_times"].get(ag["agent_id"])
        if prev_end:
            try:
                prev_dt = datetime.fromisoformat(prev_end)
                today_start = datetime.strptime(iso + f"T{minutes_to_hhmm(candidate_starts[0])}:00", "%Y-%m-%dT%H:%M:%S")
                if (today_start - prev_dt).total_seconds() < CONFIG["rest_min_minutes"] * 60:
                    return False, "insufficient_rest"
            except Exception:
                pass
        if target_dp == "night":
            max_nights = int(rules.get("hard", {}).get("max_nights_per_7d", 2))
            ct = _ledger_count_in_window(ag["agent_id"], "night", iso, 7)
            if ct >= max_nights:
                return False, "night_cap_reached"
        return True, "ok"

    scored = []
    for ag in agents:
        ok, why = eligible(ag)
        score, breakdown = agent_score(ag, r.language, r.grp, r.service, target_dp, iso)
        scored.append({
            "agent_id": ag["agent_id"], "full_name": ag["full_name"], "eligible": ok, "reason": why,
            "score": score, "breakdown": breakdown
        })
    ranked = sorted([x for x in scored if x["eligible"]], key=lambda z: z["score"], reverse=True)

    in_roster = None
    if req.agent_id:
        in_roster = next((x for x in ranked if x["agent_id"] == req.agent_id), None)

    return {
        "date": r.date,
        "filters": {"language": r.language, "grp": r.grp, "service": r.service, "target_daypart": target_dp},
        "buffers_hint": CONFIG.get("rules", {}).get("buffers", {}),
        "agent": (in_roster or next((x for x in scored if x["agent_id"] == req.agent_id), None)),
        "top_competitors": ranked[:5],
        "notes": [
            "Eligibility checks: language, group, service, site hours (optional), rest, night cap.",
            "Score = skills/services weights + fairness balance adjustment (last 7 days).",
        ]
    }

# ----------------------------------------------------------------------------- #
# NEW — M10: Free-text config parser
# ----------------------------------------------------------------------------- #
def _pct(s: str) -> float:
    return float(s) / 100.0

def _hours_to_min(s: str) -> int:
    m = re.search(r"(\d+)\s*h", s, flags=re.I)
    return int(m.group(1))*60 if m else int(s)

def _parse_dayparts(text: str) -> Dict[str, List[str]]:
    dps = {}
    for name, start, end in re.findall(r"(morning|evening|night)\s+(\d{1,2}:\d{2})\s*[-to]+\s*(\d{1,2}:\d{2})", text, flags=re.I):
        dps[name.lower()] = [start, end]
    return dps

@app.post("/config/parse")
def parse_config(req: ParseConfigRequest, apply: bool = Query(False)):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    proposed: Dict[str, Any] = {}
    rules = proposed.setdefault("rules", {}); buffers = rules.setdefault("buffers", {}); hard = rules.setdefault("hard", {})

    m = re.search(r"(service\s*level|target\s*sl)[^\d%]*?(\d{1,3})\s*%", text, flags=re.I)
    if m: proposed["target_sl"] = round(_pct(m.group(2)), 4)

    m = re.search(r"(target\s*(t|asa|answer\s*time))[^\d]*?(\d{1,4})\s*(sec|s)", text, flags=re.I)
    if m: proposed["target_t"] = int(m.group(3))

    m = re.search(r"(shrinkage)[^\d%]*?(\d{1,3})\s*%", text, flags=re.I)
    if m: proposed["shrinkage"] = round(_pct(m.group(2)), 4)

    m = re.search(r"(shift\s*(minutes|length|duration))[^\d]*?(\d{1,2})\s*(h|hour)", text, flags=re.I)
    if m: proposed["shift_minutes"] = int(m.group(3)) * 60
    m = re.search(r"(shift\s*(minutes|length|duration))[^\d]*?(\d{2,4})\s*min", text, flags=re.I)
    if m: proposed["shift_minutes"] = int(m.group(3))

    m = re.search(r"break\s*pattern[^0-9]*(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})", text, flags=re.I)
    if m: proposed["break_pattern"] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    m = re.search(r"weekday\s*buffer[^0-9%]*(\d{1,2})\s*%", text, flags=re.I)
    if m: buffers["weekday_pct"] = round(_pct(m.group(1)), 4)
    m = re.search(r"weekend\s*buffer[^0-9%]*(\d{1,2})\s*%", text, flags=re.I)
    if m: buffers["weekend_pct"] = round(_pct(m.group(1)), 4)
    m = re.search(r"(night\s*buffer|buffer\s*for\s*night)[^0-9%]*(\d{1,2})\s*%", text, flags=re.I)
    if m: buffers["night_pct"] = round(_pct(m.group(2)), 4)

    m = re.search(r"(max\s*nights?|night\s*cap)[^\d]*?(\d{1,2})\s*(per|/)\s*7", text, flags=re.I)
    if m: hard["max_nights_per_7d"] = int(m.group(2))

    m = re.search(r"(rest|minimum\s*rest)[^\d]*?(\d{1,2})\s*(h|hour)", text, flags=re.I)
    if m: proposed["rest_min_minutes"] = int(m.group(2)) * 60

    if re.search(r"(enforce\s*site\s*hours)\s*(on|true|enable)", text, flags=re.I):
        proposed["site_hours_enforced"] = True
    if re.search(r"(enforce\s*site\s*hours)\s*(off|false|disable)", text, flags=re.I):
        proposed["site_hours_enforced"] = False

    dps = _parse_dayparts(text)
    if dps:
        rules.setdefault("dayparts", {})
        rules["dayparts"].update(dps)

    if not buffers: rules.pop("buffers", None)
    if not hard: rules.pop("hard", None)
    if not rules: proposed.pop("rules", None)

    if not proposed:
        return {"applied": False, "proposed_changes": {}, "message": "No recognized settings in text."}

    if apply:
        for k, v in proposed.items():
            if k == "rules":
                cur = CONFIG.setdefault("rules", {})
                for rk, rv in v.items():
                    if isinstance(rv, dict):
                        cur.setdefault(rk, {})
                        cur[rk].update(rv)
                    else:
                        cur[rk] = rv
            else:
                CONFIG[k] = v
        try:
            import anyio
            anyio.from_thread.run(save_config_to_db)
        except Exception:
            pass
        return {"applied": True, "proposed_changes": proposed}

    return {"applied": False, "proposed_changes": proposed}

# ----------------------------------------------------------------------------- #
# CSV exports
# ----------------------------------------------------------------------------- #
@app.get("/export/requirements.csv", response_class=PlainTextResponse)
def export_requirements_csv(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
):
    rows = requirements(date=date, language=language, grp=grp, service=service)
    if not rows:
        return "date,interval_time,language,grp,service,volume,aht_sec,req_core,req_after_shrinkage,req_with_buffers\n"
    header = ["date","interval_time","language","grp","service","volume","aht_sec","req_core","req_after_shrinkage","req_with_buffers"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in header))
    return "\n".join(lines) + "\n"

@app.get("/export/roster.csv", response_class=PlainTextResponse)
def export_roster_csv(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
):
    day = generate_roster(RosterRequest(date=date, language=language, grp=grp, service=service))
    header = ["date","agent_id","full_name","site_id","shift","service","breaks"]
    lines = [",".join(header)]
    for r in day.get("roster", []):
        breaks = json.dumps(r.get("breaks") or [])
        lines.append(",".join([
            date,
            str(r.get("agent_id","")),
            str(r.get("full_name","")),
            str(r.get("site_id","")),
            str(r.get("shift","")),
            str(r.get("service","")),
            breaks.replace("\n",""),
        ]))
    return "\n".join(lines) + "\n"

# ----------------------------------------------------------------------------- #
# XLSX exports (Excel)
# ----------------------------------------------------------------------------- #
def _xlsx_response(wb, filename: str):
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return StreamingResponse(
        bio,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.get("/export/requirements.xlsx")
def export_requirements_xlsx(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
):
    from openpyxl import Workbook
    rows = requirements(date=date, language=language, grp=grp, service=service)
    wb = Workbook()
    ws = wb.active
    ws.title = "Requirements"
    header = ["date","interval_time","language","grp","service","volume","aht_sec",
              "req_core","req_after_shrinkage","req_with_buffers"]
    ws.append(header)
    for r in rows:
        ws.append([r.get(k) for k in header])
    return _xlsx_response(wb, f"requirements_{date}.xlsx")

@app.get("/export/roster.xlsx")
def export_roster_xlsx(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
):
    from openpyxl import Workbook
    day = generate_roster(RosterRequest(date=date, language=language, grp=grp, service=service))
    wb = Workbook()
    ws = wb.active
    ws.title = "Roster"
    header = ["date","agent_id","full_name","site_id","shift","service","breaks"]
    ws.append(header)
    for r in day.get("roster", []):
        ws.append([
            date,
            r.get("agent_id",""),
            r.get("full_name",""),
            r.get("site_id",""),
            r.get("shift",""),
            r.get("service",""),
            json.dumps(r.get("breaks") or [])
        ])
    return _xlsx_response(wb, f"roster_{date}.xlsx")

# ----------------------------------------------------------------------------- #
# Quick roll-up: total required (with buffers) by date and daypart
# ----------------------------------------------------------------------------- #
@app.get("/reports/requirements/summary")
def requirements_summary(
    date_from: str = Query(..., description="dd-mm-yyyy"),
    date_to: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    service: Optional[str] = None,
):
    iso_from = parse_ddmmyyyy(date_from)
    iso_to = parse_ddmmyyyy(date_to)
    cur = datetime.strptime(iso_from, "%Y-%m-%d")
    end = datetime.strptime(iso_to, "%Y-%m-%d")
    rules = CONFIG.get("rules", {})
    dp_bounds = get_daypart_bounds(rules.get("dayparts", {}))
    out = []
    while cur <= end:
        date_dd = to_ddmmyyyy(cur.strftime("%Y-%m-%d"))
        reqs = requirements(date=date_dd, language=language, grp=grp, service=service)
        # bucket by inferred daypart from interval_time
        buckets = {"morning": 0, "evening": 0, "night": 0}
        for r in reqs:
            iv = str(r["interval_time"])
            mins = hhmm_to_minutes(iv)
            dp = infer_daypart(mins, dp_bounds)
            buckets[dp] = buckets.get(dp, 0) + int(r["req_with_buffers"])
        out.append({"date": date_dd, **buckets, "total": sum(buckets.values())})
        cur += timedelta(days=1)
    return out


# ----------------------------------------------------------------------------- #
# Playground — dd-mm-yyyy (unchanged UI fields)
# ----------------------------------------------------------------------------- #
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
  <div><label>Service</label><input id="svc"  value=""></div>
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
    service: document.getElementById('svc').value,
    shrinkage: parseFloat(document.getElementById('shr').value),
    target_sl: parseFloat(document.getElementById('tsl').value),
    target_t: parseInt(document.getElementById('tt').value,10),
  }};
}}
function runReq() {{
  const v = vals();
  const q = new URLSearchParams({{ 
    date: v.date, language: v.language, grp: v.grp, service: v.service,
    target_sl: v.target_sl, target_t: v.target_t, shrinkage: v.shrinkage 
  }});
  fetch('/requirements?'+q).then(async r=>{{ const t=await r.text(); try{{out.textContent=JSON.stringify(JSON.parse(t),null,2)}}catch(e){{out.textContent=t}} }})
    .catch(e=>out.textContent=String(e));
}}
function runRoster() {{
  const v = vals();
  fetch('/generate-roster', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(v) }})
    .then(async r=>{{ const t=await r.text(); try{{out.textContent=JSON.stringify(JSON.parse(t),null,2)}}catch(e){{out.textContent=t}} }})
    .catch(e=>out.textContent=String(e));
}}
function runRange() {{
  const v = vals();
  const body = {{ date_from: document.getElementById('from').value,
                  date_to: document.getElementById('to').value,
                  language: v.language, grp: v.grp, service: v.service, persist: true }};
  fetch('/generate-roster-range', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body) }})
    .then(async r=>{{ const t=await r.text(); try{{out.textContent=JSON.stringify(JSON.parse(t),null,2)}}catch(e){{out.textContent=t}} }})
    .catch(e=>out.textContent=String(e));
}}
</script>
</body></html>
"""
    return HTMLResponse(html)
# --- M12: My Week -------------------------------------------------------------

@app.get("/me/week")
def my_week(
    agent_id: str = Query(...),
    date_from: str = Query(..., description="dd-mm-yyyy"),
    date_to: str = Query(..., description="dd-mm-yyyy"),
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    iso_from = parse_ddmmyyyy(date_from)
    iso_to   = parse_ddmmyyyy(date_to)
    rows = sb_get("rosters", {
        "select": "date,agent_id,full_name,site_id,shift,breaks,meta",
        "agent_id": f"eq.{agent_id}",
        "date": f"gte.{iso_from}",
        "order": "date.asc"
    })
    # collapse to latest per day (client-side)
    by_day = {}
    for r in rows:
        if r["date"] <= iso_to:
            by_day[r["date"]] = r  # last one wins due to order
    out = []
    for iso, r in sorted(by_day.items()):
        r["date"] = to_ddmmyyyy(r["date"])
        out.append(r)
    return {"agent_id": agent_id, "from": date_from, "to": date_to, "days": out}


# --- M12: AI swap suggestions --------------------------------------------------

class SwapSuggestIn(BaseModel):
    date: str            # dd-mm-yyyy
    agent_id: str
    k: Optional[int] = 5

@app.post("/suggest/swaps")
def suggest_swaps(body: SwapSuggestIn):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    iso = parse_ddmmyyyy(body.date)

    # the agent's roster for that day (must exist)
    me = _get_roster_row(iso, body.agent_id)
    if not me:
        raise HTTPException(404, "No saved roster for this agent on that date.")
    me_start = hhmm_to_minutes(me["shift"].split("-")[0].strip())
    # infer target daypart using configured dayparts
    dp_bounds = get_daypart_bounds(CONFIG.get("rules", {}).get("dayparts", {}))
    me_dp = infer_daypart(me_start, dp_bounds) if dp_bounds else "morning"

def _shift_minutes(shift_str: str) -> Tuple[int, int]:
    """'HH:MM - HH:MM' -> (start_min, end_min) with end possibly < start if crosses midnight."""
    st_str, en_str = [s.strip() for s in shift_str.split("-")]
    st_h, st_m = map(int, st_str.split(":"))
    en_h, en_m = map(int, en_str.split(":"))
    st = st_h * 60 + st_m
    en = en_h * 60 + en_m
    return st, en

    # fetch all rostered that day
    all_rows = sb_get("rosters", {
        "select": "date,agent_id,full_name,site_id,shift,breaks,meta",
        "date": f"eq.{iso}",
        "order": "agent_id.asc"
    })
    # collapse to latest per (date, agent_id)
    seen = set()
    rostered = []
    for r in all_rows:
        key = (r["date"], r["agent_id"])
        if key in seen: 
            continue
        seen.add(key)
        rostered.append(r)

    # load agents master
    sel = "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services"
    agents = {a["agent_id"]: a for a in sb_get("agents", {"select": sel, "limit": 2000})}
    for a in agents.values():
        a["trained_groups"] = ensure_list(a.get("trained_groups"))
        a["trained_services"] = ensure_list(a.get("trained_services"))

    # the filters used to assign me (if any exist in meta)
    lang = (me.get("meta") or {}).get("service_language") or (me.get("meta") or {}).get("language")
    grp  = (me.get("meta") or {}).get("grp")
    svc  = (me.get("meta") or {}).get("service")

    # build candidate list = everyone else rostered same day
    candidates = [r for r in rostered if r.get("agent_id") != body.agent_id]

    def can_cover(agent_id: str, target_row: Dict[str, Any]) -> bool:
        ag = agents.get(agent_id)
        if not ag: 
            return False
        svc_need = (target_row.get("meta") or {}).get("service")
        if svc_need:
            trained = ensure_list(ag.get("trained_services"))
            if trained and svc_need not in trained:
                return False
        # optional: language/group checks using target_row.meta
        lang_need = (target_row.get("meta") or {}).get("language")
        if lang_need and not (ag.get("primary_language")==lang_need or ag.get("secondary_language")==lang_need):
            return False
        grp_need = (target_row.get("meta") or {}).get("grp")
        if grp_need and grp_need not in ensure_list(ag.get("trained_groups")):
            return False
        return True

    # score: how "fair" + skill aligned a swap would be for both directions
    iso_day = iso
    ideas = []
    for other in candidates:
        other_id = other["agent_id"]
        if not (can_cover(other_id, me) and can_cover(body.agent_id, other)):
            continue

        # eligibility checks similar to /explain-assignment fairness
        # compute delta fairness: if me takes other's start, and other takes mine
        other_start = hhmm_to_minutes(other["shift"].split("-")[0].strip())
        other_dp = infer_daypart(other_start, dp_bounds) if dp_bounds else "morning"

        # simple “benefit” heuristic: prefer swaps that reduce repeated dayparts in last 7d
        me_rep   = _ledger_count_in_window(body.agent_id, me_dp, iso_day, 7)
        other_rep= _ledger_count_in_window(other_id, other_dp, iso_day, 7)

        # pretend post-swap: me gets other_dp, other gets me_dp
        me_post_rep    = _ledger_count_in_window(body.agent_id, other_dp, iso_day, 7)
        other_post_rep = _ledger_count_in_window(other_id, me_dp, iso_day, 7)

        benefit = 0.0
        if me_rep >= 2 and other_dp != me_dp: benefit += 1.0
        if other_rep >= 2 and me_dp != other_dp: benefit += 1.0

        # skill alignment via your agent_score for each direction
        me_ag   = agents.get(body.agent_id, {})
        other_ag= agents.get(other_id, {})
        s1,_ = agent_score(me_ag, lang, grp, svc, other_dp, iso_day)
        s2,_ = agent_score(other_ag, lang, grp, svc, me_dp, iso_day)  # coarse reuse

        total_score = benefit + 0.25*(s1 + s2)

        ideas.append({
            "swap_with": other_id,
            "swap_with_name": other.get("full_name"),
            "their_shift": other.get("shift"),
            "your_shift": me.get("shift"),
            "benefit_hint": {"you_relief": me_rep, "them_relief": other_rep},
            "score": round(total_score, 3)
        })

    ideas.sort(key=lambda x: x["score"], reverse=True)
    return {"date": body.date, "agent_id": body.agent_id, "suggestions": ideas[:max(1, body.k or 5)]}


# --- M12: Preferences (optional, simple) --------------------------------------

class PrefsIn(BaseModel):
    agent_id: str
    prefer_dayparts: Optional[List[str]] = None
    avoid_dayparts: Optional[List[str]] = None
    blackout_dates: Optional[List[str]] = None  # dd-mm-yyyy

@app.get("/prefs")
def prefs_get(agent_id: str = Query(...)):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    rows = sb_get("agent_prefs", {"select": "*", "agent_id": f"eq.{agent_id}", "limit": 1})
    return rows[0] if rows else {"agent_id": agent_id, "prefer_dayparts": [], "avoid_dayparts": [], "blackout_dates": []}

@app.post("/prefs")
def prefs_set(body: PrefsIn):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    blk = []
    for d in (body.blackout_dates or []):
        blk.append(parse_ddmmyyyy(d))
    row = {
        "agent_id": body.agent_id,
        "prefer_dayparts": body.prefer_dayparts or [],
        "avoid_dayparts": body.avoid_dayparts or [],
        "blackout_dates": blk,
        "updated_at": datetime.utcnow().isoformat()
    }
    sb_upsert("agent_prefs", [row], on_conflict="agent_id")
    return {"status": "ok"}

# ----------------------------------------------------------------------------- #
# NEW — Convenience: read persisted rosters by date range (for verification)
# ----------------------------------------------------------------------------- #
@app.get("/rosters")
def rosters(
    date_from: str = Query(..., description="dd-mm-yyyy"),
    date_to: str = Query(..., description="dd-mm-yyyy"),
    agent_id: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    iso_from = parse_ddmmyyyy(date_from)
    iso_to = parse_ddmmyyyy(date_to)
    params = {
        "select": "date,agent_id,full_name,site_id,shift,breaks,meta",
        "date": f"gte.{iso_from}",
        "order": "date.asc,agent_id.asc"
    }
    # Supabase REST doesn't support combined gte/lte in one param; chain 'and' via 'and='
    # Simpler: filter lte after fetch; or do two filters with 'and' … for brevity, fetch window and filter client-side.
    rows = sb_get("rosters", params)
    out = []
    for r in rows:
        if r["date"] <= iso_to and (not agent_id or r.get("agent_id")==agent_id):
            r["date"] = to_ddmmyyyy(r["date"])
            out.append(r)
    return out

# ----------------------------------------------------------------------------- #
# NEW — M11: Shift swap requests + approvals + commit
# ----------------------------------------------------------------------------- #
def _iso(d_ddmmyyyy: str) -> str:
    return parse_ddmmyyyy(d_ddmmyyyy)

def _get_roster_row(iso_date: str, agent_id: str) -> Optional[Dict[str, Any]]:
    rows = sb_get("rosters", {
        "select": "date,agent_id,full_name,site_id,shift,breaks,meta",
        "date": f"eq.{iso_date}",
        "agent_id": f"eq.{agent_id}",
        "limit": 1
    })
    return rows[0] if rows else None

def _load_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    sel = "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services"
    rows = sb_get("agents", {"select": sel, "agent_id": f"eq.{agent_id}", "limit": 1})
    if rows:
        rows[0]["trained_groups"] = ensure_list(rows[0].get("trained_groups"))
        rows[0]["trained_services"] = ensure_list(rows[0].get("trained_services"))
        return rows[0]
    return None

def _check_capability_for_swap(agent: Dict[str, Any], target_meta: Dict[str, Any]) -> Tuple[bool, str]:
    # Minimal: verify service coverage if service present in meta
    svc = (target_meta or {}).get("service")
    if svc:
        trained = ensure_list(agent.get("trained_services"))
        if trained and svc not in trained:
            return False, "service_not_trained"
    return True, "ok"

def _apply_swap_to_db(iso_date: str, a_from: str, a_to: str):
    """Swap the shift strings between two agents; keep each row's own breaks/meta."""
    r_from = _get_roster_row(iso_date, a_from)
    r_to   = _get_roster_row(iso_date, a_to)
    if not r_from or not r_to:
        raise HTTPException(400, "One or both agents have no persisted roster on this date.")

    new_from = dict(r_from)
    new_to   = dict(r_to)
    new_from["shift"], new_to["shift"] = r_to.get("shift"), r_from.get("shift")

    # Keep meta; if you want to also swap meta.service, uncomment below:
    # mf = dict(new_from.get("meta") or {}); mt = dict(new_to.get("meta") or {})
    # mf_svc, mt_svc = mf.get("service"), mt.get("service")
    # mf["service"], mt["service"] = mt_svc, mf_svc
    # new_from["meta"] = mf; new_to["meta"] = mt

    # Persist as new rows (historical trail). If you want hard updates, add a PK and use upsert.
    sb_post("rosters", [new_from, new_to])
# --- Swap audit helpers -------------------------------------------------------
def _audit_swap(decision: str, req: Dict[str, Any], before_from: Dict[str, Any], before_to: Dict[str, Any],
                after_from: Dict[str, Any], after_to: Dict[str, Any], approver: str, notes: Optional[Dict[str, Any]] = None):
    try:
        row = {
            "swap_id": req.get("id"),
            "date": req.get("date"),
            "agent_from": req.get("agent_from"),
            "agent_to": req.get("agent_to"),
            "before_from_shift": (before_from or {}).get("shift"),
            "before_to_shift": (before_to or {}).get("shift"),
            "after_from_shift": (after_from or {}).get("shift"),
            "after_to_shift": (after_to or {}).get("shift"),
            "approver": approver,
            "decision": decision,  # approved / denied
            "notes": notes or {},
        }
        sb_post("swaps_audit", [row])
    except Exception as e:
        log.info(f"[swap audit] skipped: {e}")

@app.get("/swap/list")
def swap_list(
    status: Optional[str] = Query(None),
    date: Optional[str] = Query(None, description="dd-mm-yyyy"),
    agent_id: Optional[str] = None,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    params = {"select": "*", "order": "created_at.desc"}
    if status:
        params["status"] = f"eq.{status}"
    if date:
        params["date"] = f"eq.{parse_ddmmyyyy(date)}"
    if agent_id:
        params["or"] = f"(agent_from.eq.{agent_id},agent_to.eq.{agent_id})"
    return sb_get("swap_requests", params)

@app.post("/swap/request")
def swap_request(body: SwapRequestIn):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    iso = _iso(body.date)

    r_from = _get_roster_row(iso, body.agent_from)
    r_to   = _get_roster_row(iso, body.agent_to)
    if not r_from or not r_to:
        raise HTTPException(400, "Both agents need an existing roster saved for this date.")

    ag_from = _load_agent(body.agent_from) or {}
    ag_to   = _load_agent(body.agent_to) or {}
    ok_to, why_to     = _check_capability_for_swap(ag_to,   r_from.get("meta") or {})
    ok_from, why_from = _check_capability_for_swap(ag_from, r_to.get("meta") or {})
    if not ok_to:
        raise HTTPException(400, f"agent_to cannot cover agent_from shift: {why_to}")
    if not ok_from:
        raise HTTPException(400, f"agent_from cannot cover agent_to shift: {why_from}")

    # Create as PROPOSED; waits for counter-party decision
    record = {
        "date": iso,
        "agent_from": body.agent_from,
        "agent_to": body.agent_to,
        "status": "proposed",
        "reason": body.reason or "",
        "meta": {"from": r_from, "to": r_to}
    }
    sb_post("swap_requests", [record])

    # Optional “force” = immediate apply (admin tools)
    applied = False
    if body.force:
        _apply_swap_to_db(iso, body.agent_from, body.agent_to)
        applied = True

    return {"status": "ok", "created": True, "auto_applied": applied}

@app.post("/swap/respond")
@require_role("agent", "supervisor", "admin")
async def swap_respond(body: SwapRespondIn, request: Request):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # Load request
    rows = sb_get("swap_requests", {"select": "*", "id": f"eq.{body.id}", "limit": 1})
    if not rows:
        raise HTTPException(404, "Swap request not found")
    req = rows[0]
    if req.get("status") != "proposed":
        raise HTTPException(400, f"Swap is {req.get('status')}, not proposed")

    # Only the counter-party (agent_to) is allowed to respond
    if body.actor_agent_id != req.get("agent_to"):
        raise HTTPException(403, "Only the counter-party can respond to this swap")

    update = dict(req)
    meta = dict(update.get("meta") or {})
    meta["counterparty_notes"] = body.notes or ""
    update["meta"] = meta

    if body.decision.lower() == "accept":
        update["status"] = "accepted"
    elif body.decision.lower() == "decline":
        update["status"] = "declined_by_counterparty"
    else:
        raise HTTPException(400, "decision must be 'accept' or 'decline'")

    sb_upsert("swap_requests", [update], on_conflict="id")
    return {"status": "ok", "id": body.id, "new_status": update["status"]}


@app.post("/swap/approve")
@require_role("scheduler", "supervisor", "admin")
async def swap_approve(body: SwapDecisionIn, request: Request):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    approver = _user_from_headers(request) or (body.approver or "approver")

    reqs = sb_get("swap_requests", {"select": "*", "id": f"eq.{body.id}", "limit": 1})
    if not reqs:
        raise HTTPException(404, "Swap request not found")
    req = reqs[0]
    if req.get("status") != "accepted":
        raise HTTPException(400, f"Supervisor can only approve 'accepted' requests (current: {req.get('status')})")

    iso   = req["date"]; a_from = req["agent_from"]; a_to = req["agent_to"]

    # Before/after snapshots
    r_from_before = _get_roster_row(iso, a_from)
    r_to_before   = _get_roster_row(iso, a_to)
    if not r_from_before or not r_to_before:
        raise HTTPException(400, "Both agents need an existing roster saved for this date.")

    # Rule checks (rest minimum; optional weekly caps if configured)
    hard = CONFIG.get("rules", {}).get("hard", {})
    min_rest = int(hard.get("swap_min_rest_minutes") or CONFIG.get("rest_min_minutes", 12*60))

    # New shifts after swap
    af_st, af_en = _shift_minutes(r_from_before["shift"])
    at_st, at_en = _shift_minutes(r_to_before["shift"])

    # Compare to previous-day end marks if you persist them
    prev_end_from = CONFIG.get("prev_end_times", {}).get(a_from)
    prev_end_to   = CONFIG.get("prev_end_times", {}).get(a_to)

    def ok_rest(prev_end_iso: Optional[str], start_min: int) -> bool:
        if not prev_end_iso:
            return True
        try:
            prev_dt = datetime.fromisoformat(prev_end_iso)
            # start of THIS day at the swapped start minute
            today_start = datetime.strptime(iso + f"T{minutes_to_hhmm(start_min)}:00", "%Y-%m-%dT%H:%M:%S")
            if (today_start - prev_dt).total_seconds() < min_rest * 60:
                return False
        except Exception:
            return True
        return True

    # After swap, a_from would work a_to's shift start; a_to would work a_from's start
    if not ok_rest(prev_end_from, at_st) or not ok_rest(prev_end_to, af_st):
        raise HTTPException(400, "Swap violates minimum rest period")

    # (Optional) TODO: check weekly hours/days caps here if you later set the values in CONFIG

    # Apply
    _apply_swap_to_db(iso, a_from, a_to)

    # After snapshots
    r_from_after = _get_roster_row(iso, a_from)
    r_to_after   = _get_roster_row(iso, a_to)

    # Mark request approved
    update = dict(req)
    update["status"] = "approved"
    update["decision_by"] = approver
    update["decision_at"] = datetime.utcnow().isoformat()
    sb_upsert("swap_requests", [update], on_conflict="id")

    # Audit
    _audit_swap("approved", req, r_from_before, r_to_before, r_from_after, r_to_after, approver,
                notes=(body.notes and {"notes": body.notes}))
    return {"status": "ok", "approved": True, "id": body.id}



@app.post("/swap/deny")
@require_role("scheduler", "supervisor", "admin")
async def swap_deny(body: SwapDecisionIn, request: Request):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    approver = _user_from_headers(request) or (body.approver or "approver")

    reqs = sb_get("swap_requests", {"select": "*", "id": f"eq.{body.id}", "limit": 1})
    if not reqs:
        raise HTTPException(404, "Swap request not found")
    req = reqs[0]
    if req.get("status") != "pending":
        raise HTTPException(400, f"Swap request is {req.get('status')}, not pending")

    update = dict(req)
    update["status"] = "denied"
    update["decision_by"] = approver
    update["decision_at"] = datetime.utcnow().isoformat()
    if body.notes:
        m = dict(update.get("meta") or {})
        m["decision_notes"] = body.notes
        update["meta"] = m
    sb_upsert("swap_requests", [update], on_conflict="id")

    # Audit log (no before/after diff because no change)
    _audit_swap("denied", req, None, None, None, None, approver, notes=(body.notes and {"notes": body.notes}))

    return {"status": "ok", "denied": True, "id": body.id}

