# main.py — CCC Scheduler API (Supabase-backed, dd-mm-yyyy I/O) — M5 consolidated

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx, json

# ============================================================================
# Supabase REST setup (READ via ANON; WRITE via SERVICE_ROLE if provided)
# ============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")  # public/anon
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # server-side writes (optional)

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
    # Prefer service key for writes; otherwise fall back to anon (with risk)
    _write_token = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    HEADERS_WRITE = {
        "apikey": _write_token,
        "Authorization": f"Bearer {_write_token}",
    }

# ============================================================================
# App + CORS
# ============================================================================
app = FastAPI(title="CCC Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Helpers — dates & time (UI uses dd-mm-yyyy; DB uses yyyy-mm-dd)
# ============================================================================
def parse_ddmmyyyy(d: str) -> str:
    """Input dd-mm-yyyy -> output yyyy-mm-dd (ISO, for DB)."""
    try:
        return datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, f"Invalid date '{d}'. Use dd-mm-yyyy.")

def to_ddmmyyyy(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%m-%Y")

def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ============================================================================
# CONFIG — default (will be merged with DB row at runtime)
# DB table: public.wfm_config, single row id=1 with column "config" (jsonb)
# ============================================================================
CONFIG_DEFAULT: Dict[str, Any] = {
    # requirements
    "interval_seconds": 1800,
    "target_sl": 0.80,
    "target_t": 20,
    "shrinkage": 0.30,

    # scheduling
    "shift_minutes": 9 * 60,      # 9h
    "break_pattern": [15, 30, 15],
    "break_cap_frac": 0.25,
    "no_head": 60,
    "no_tail": 60,
    "lunch_gap": 120,

    # staffing constraints
    "site_hours_enforced": False,
    "site_hours": {},             # e.g., {"QA":{"open":"10:00","close":"19:00","tz":"Asia/Qatar"}}
    "rest_min_minutes": 12 * 60,
    "prev_end_times": {},         # { "A001": "YYYY-MM-DDTHH:MM:SS" }

    # default/global timezone label (informational)
    "timezone": "UTC",

    # scoring/priority knobs
    "weight_primary_language": 2.0,
    "weight_secondary_language": 1.0,
    "weight_group_exact": 1.5,
    "weight_service_match": 1.0,
}

CONFIG: Dict[str, Any] = dict(CONFIG_DEFAULT)

async def load_config_from_db():
    """Merge DB config (if exists) into CONFIG."""
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
        # non-fatal
        pass

async def save_config_to_db():
    """Upsert CONFIG to DB (row id=1)."""
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

# ============================================================================
# Health
# ============================================================================
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

# ============================================================================
# Erlang C & Requirements math
# ============================================================================
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
    lam = volume / interval_seconds      # arrivals/sec
    mu = 1.0 / max(aht_sec, 1)           # service rate/agent
    a = lam / mu                         # offered load
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

# ============================================================================
# DB helpers
# ============================================================================
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

# ============================================================================
# Thin list endpoints
# ============================================================================
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
        # primary or secondary
        params["or"] = f"(primary_language.eq.{language},secondary_language.eq.{language})"
    if grp:
        params["trained_groups"] = f"cs.{{{grp}}}"  # PostgREST @> array contains
    return sb_get("agents", params)

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
        "order": "interval_time.asc,language.asc,grp.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"
    rows = sb_get("forecasts", params)
    for row in rows:
        row["date"] = to_ddmmyyyy(row["date"])
    return rows

# ============================================================================
# Requirements (per-interval)
# ============================================================================
@app.get("/requirements")
def requirements(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
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
        "order": "interval_time.asc,language.asc,grp.asc",
    }
    if language:
        params["language"] = f"eq.{language}"
    if grp:
        params["grp"] = f"eq.{grp}"

    rows = sb_get("forecasts", params)

    out: List[Dict[str, Any]] = []
    for row in rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = N_core if shrinkage >= 0.999 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
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
        })
    return out

# ============================================================================
# Models
# ============================================================================
class RosterRequest(BaseModel):
    date: str                    # dd-mm-yyyy
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None

class RangeRequest(BaseModel):
    date_from: str               # dd-mm-yyyy
    date_to: str                 # dd-mm-yyyy
    language: Optional[str] = None
    grp: Optional[str] = None
    persist: Optional[bool] = False  # if true, auto-save each day's roster

class SaveRosterRequest(BaseModel):
    date: str                    # dd-mm-yyyy
    roster: List[Dict[str, Any]]

# ============================================================================
# Scoring — pick best agents by skill/priority
# ============================================================================
def agent_score(agent: Dict[str, Any], language: Optional[str], grp: Optional[str]) -> float:
    score = 0.0
    if language:
        if agent.get("primary_language") == language:
            score += CONFIG.get("weight_primary_language", 2.0)
        if agent.get("secondary_language") == language:
            score += CONFIG.get("weight_secondary_language", 1.0)
    if grp and grp in (agent.get("trained_groups") or []):
        score += CONFIG.get("weight_group_exact", 1.5)
    # Service awareness (if present)
    # We don't pass service now; this hook remains for later extensibility.
    return score

# ============================================================================
# Generate roster (single day)
# ============================================================================
@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # resolve config overrides (fall back to CONFIG)
    interval_seconds = req.interval_seconds or CONFIG["interval_seconds"]
    target_sl = req.target_sl if req.target_sl is not None else CONFIG["target_sl"]
    target_t = req.target_t if req.target_t is not None else CONFIG["target_t"]
    shrinkage = req.shrinkage if req.shrinkage is not None else CONFIG["shrinkage"]

    iso = parse_ddmmyyyy(req.date)

    # forecasts for the day
    f_params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{iso}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
    }
    fc_rows = sb_get("forecasts", f_params)

    intervals = []
    for row in fc_rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        req_after = N_core if shrinkage >= 0.999 else ceil(N_core / (1.0 - max(0.0, min(shrinkage, 0.95))))
        intervals.append({"t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after})

    # agents
    sel = "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services"
    agents = sb_get("agents", {"select": sel, "limit": 2000})

    # capability filter
    if req.language or req.grp:
        pool = [ag for ag in agents if
                (not req.language or (ag.get("primary_language") == req.language or ag.get("secondary_language") == req.language)) and
                (not req.grp or (req.grp in (ag.get("trained_groups") or [])))]
    else:
        pool = agents[:]

    # sort pool by score (desc)
    pool.sort(key=lambda ag: agent_score(ag, req.language, req.grp), reverse=True)

    # demand grid
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
                # site hours rule (optional, requires whole shift within window)
                if CONFIG["site_hours_enforced"]:
                    site = ag.get("site_id")
                    sh = CONFIG["site_hours"].get(site) if site else None
                    if sh and sh.get("open") and sh.get("close"):
                        o = hhmm_to_minutes(sh["open"])
                        c = hhmm_to_minutes(sh["close"])
                        if not (o <= start_min and end_min <= c):
                            continue
                # cross-day rest
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
                "date": req.date,  # dd-mm-yyyy for output
                "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                "notes": "auto assignment",
            })
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1
            if all(assigned[t] >= demand[t] for t in times):
                break

    # Break planning (fairness cap; avoid peak stress)
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
            # minimal violation fallback
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
            # 60-min window around center
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
            if end_min <= start_min:
                end_min += 24 * 60

            NO_HEAD, NO_TAIL, GAP = CONFIG["no_head"], CONFIG["no_tail"], CONFIG["lunch_gap"]
            place_start = start_min + NO_HEAD
            place_end   = end_min   - NO_TAIL
            pattern = CONFIG["break_pattern"]
            if place_end - place_start < (sum(pattern) + 2 * GAP):
                item["breaks"] = []
                continue

            mid = (start_min + end_min) // 2
            lunch_target = clamp(mid, place_start + GAP // 2, place_end - GAP // 2)

            # lunch first (pattern[1] assumed long)
            lunch_dur = max(10, int(pattern[1]))
            lunch_cands = window_candidates(lunch_target, lunch_dur, place_start, place_end)
            lunch_s = choose_slot(lunch_cands, lunch_dur) or snap_to_grid(lunch_target)
            lunch_e = lunch_s + lunch_dur
            for t in time_list:
                if lunch_s <= t < lunch_e:
                    break_load[t] += 1

            # small break before lunch
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

            # small break after lunch
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
            "Greedy assignment; fairness cap; site hours (optional); 12h cross-day rest via prev_end_times.",
            "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes.",
        ],
    }
    return summary

# ============================================================================
# Generate roster for a range (+ optional persist)
# ============================================================================
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
        day = generate_roster(day_req)
        out[cur.strftime("%Y-%m-%d")] = day

        if req.persist and day.get("roster"):
            # convert to DB rows and save
            iso = cur.strftime("%Y-%m-%d")
            rows = []
            for item in day["roster"]:
                rows.append({
                    "date": iso,
                    "agent_id": item.get("agent_id"),
                    "full_name": item.get("full_name"),
                    "site_id": item.get("site_id"),
                    "shift": item.get("shift"),
                    "breaks": item.get("breaks") or [],
                    "meta": item,
                })
            if rows:
                sb_post("rosters", rows)

        cur += timedelta(days=1)
    return out

# ============================================================================
# Save roster (single day) to Supabase
# ============================================================================
@app.post("/save-roster")
def save_roster(req: SaveRosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    if not isinstance(req.roster, list):
        raise HTTPException(400, "roster must be a list")
    iso = parse_ddmmyyyy(req.date)
    rows = []
    for item in req.roster:
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
    return {"status": "ok", "inserted": len(rows)}

# ============================================================================
# Runtime config endpoints (DB-backed)
# ============================================================================
@app.get("/config")
def get_config():
    return CONFIG

@app.put("/config")
def put_config(body: Dict[str, Any] = Body(...)):
    # allow partial updates
    for k, v in body.items():
        if k in CONFIG_DEFAULT:
            CONFIG[k] = v
    # persist to DB
    try:
        # fire-and-forget save (raise means hard fail)
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
    # replace only known keys
    applied = {k: new_conf[k] for k in new_conf if k in CONFIG_DEFAULT}
    CONFIG.update(applied)
    # persist to DB
    try:
        import anyio
        anyio.from_thread.run(save_config_to_db)
    except Exception:
        pass
    return {"status": "ok", "applied_keys": sorted(applied.keys())}

# ============================================================================
# CSV exports
# ============================================================================
@app.get("/export/requirements.csv", response_class=PlainTextResponse)
def export_requirements_csv(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    rows = requirements(date=date, language=language, grp=grp)
    if not rows:
        return "date,interval_time,language,grp,service,volume,aht_sec,req_core,req_after_shrinkage\n"
    header = ["date","interval_time","language","grp","service","volume","aht_sec","req_core","req_after_shrinkage"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in header))
    return "\n".join(lines) + "\n"

@app.get("/export/roster.csv", response_class=PlainTextResponse)
def export_roster_csv(
    date: str = Query(..., description="dd-mm-yyyy"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
):
    # Build the roster using the same filters the UI uses
    day = generate_roster(RosterRequest(date=date, language=language, grp=grp))

    header = ["date","agent_id","full_name","site_id","shift","breaks"]
    lines = [",".join(header)]

    # If there’s no roster (no demand or no matching forecasts), still return a valid CSV
    for r in day.get("roster", []):
        breaks = json.dumps(r.get("breaks") or []).replace("\n", "")
        lines.append(",".join([
            date,
            str(r.get("agent_id","")),
            str(r.get("full_name","")),
            str(r.get("site_id","")),
            str(r.get("shift","")),
            breaks,
        ]))

    if len(lines) == 1:
        # header only
        return ",".join(header) + "\n"

    return "\n".join(lines) + "\n"


# ============================================================================
# Playground — dd-mm-yyyy
# ============================================================================
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
                  language: v.language, grp: v.grp, persist: true }};
  fetch('/generate-roster-range', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body) }})
    .then(r=>r.json()).then(j=>out.textContent=JSON.stringify(j,null,2))
    .catch(e=>out.textContent=String(e));
}}
</script>
</body></html>
"""
    return HTMLResponse(html)
