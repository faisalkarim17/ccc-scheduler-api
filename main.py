from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import os, httpx
from math import factorial, exp, ceil
from datetime import datetime, timedelta, timezone

# -----------------------------------------------------------------------------
# ENV / Supabase REST setup
# -----------------------------------------------------------------------------
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
    }

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="CCC Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # you can restrict this to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def time_to_minutes(tstr: str) -> int:
    # interval_time is "HH:MM:SS" in Supabase; we ignore seconds
    hh, mm, *_ = tstr.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Erlang C
def erlang_c(a: float, N: int) -> float:
    if N <= a:  # stability guard
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
        p_wait_gt_t = Ec * exp(-(N * mu - lam) * target_t)
        sl = 1.0 - p_wait_gt_t
        if sl >= target_sl:
            return N
        N += 1
    return N

# -----------------------------------------------------------------------------
# Site hours (simple config; edit as you like)
# times are minutes from midnight local; shifts must be fully inside these windows
# -----------------------------------------------------------------------------
SITE_HOURS: Dict[str, Dict[str, int]] = {
    # site_id: {"open": minutes, "close": minutes}
    "QA": {"open":  9 * 60, "close": 18 * 60},  # 09:00–18:00
    "IN": {"open":  9 * 60, "close": 18 * 60},
    "CN": {"open":  9 * 60, "close": 18 * 60},
    "EU": {"open":  9 * 60, "close": 18 * 60},
    # default handled below (if site not present)
}

def site_allows(site_id: Optional[str], start_min: int, end_min: int) -> bool:
    conf = SITE_HOURS.get(site_id or "", None)
    if conf is None:
        # default window if site not configured
        conf = {"open": 8 * 60, "close": 20 * 60}  # 08:00–20:00
    return (start_min >= conf["open"]) and (end_min <= conf["close"])

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RosterRequest(BaseModel):
    date: str                    # YYYY-MM-DD (local)
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: float = 0.30
    interval_seconds: int = 1800
    target_sl: float = 0.80
    target_t: int = 20

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
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
    interval_seconds: int = 1800,
    target_sl: float = 0.80,
    target_t: int = 20,
    shrinkage: float = 0.30,
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{language}"} if language else {}),
        **({"grp": f"eq.{grp}"}         if grp else {}),
    }, timeout=30)
    r.raise_for_status()
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
# Generate roster (with breaks, fairness, rest across days, site hours)
# -----------------------------------------------------------------------------
@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # 1) Load forecasts (for the day, optionally filtered)
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{req.date}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"}         if req.grp else {}),
    }, timeout=30)
    r.raise_for_status()
    fc_rows = r.json()

    # Build interval requirements using the same math as /requirements
    intervals: List[Dict[str, Any]] = []
    for row in fc_rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, req.interval_seconds, req.target_sl, req.target_t)
        req_after = N_core if req.shrinkage >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(req.shrinkage, 0.95))))
        intervals.append({
            "t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after
        })

    # 2) Load agents
    a = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params={
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": 2000
    }, timeout=30)
    a.raise_for_status()
    agents: List[Dict[str, Any]] = a.json()

    # 3) Load last_end_utc to enforce 12-hour rest across days
    #    Table must exist: agent_state(agent_id text primary key, last_end_utc timestamptz)
    last_end_map: Dict[str, Optional[datetime]] = {}
    try:
        s = httpx.get(
            f"{REST_BASE}/agent_state",
            headers=HEADERS,
            params={"select": "agent_id,last_end_utc"},
            timeout=15,
        )
        if s.status_code == 200:
            for row in s.json():
                try:
                    last_end_map[row["agent_id"]] = datetime.fromisoformat(row["last_end_utc"].replace("Z", "+00:00"))
                except Exception:
                    last_end_map[row["agent_id"]] = None
    except Exception:
        # fail-soft: no rest restriction if table missing
        last_end_map = {}

    # 4) Build time grid, demand & assigned
    times = sorted({ time_to_minutes(x["t"]) for x in intervals })
    times_map: Dict[int, List[Dict[str, Any]]] = {t: [iv for iv in intervals if time_to_minutes(iv["t"]) == t] for t in times}
    demand:   Dict[int, int] = {t: sum(iv["req"] for iv in times_map[t]) for t in times}
    assigned: Dict[int, int] = {t: 0 for t in times}

    # Filters: language/group capability
    def can_cover(agent, lang, grp):
        lang_ok = (agent["primary_language"] == lang) or (agent["secondary_language"] == lang)
        grp_ok = grp in (agent.get("trained_groups") or [])
        return bool(lang_ok and grp_ok)

    if req.language or req.grp:
        pool = [ag for ag in agents if all([
            (not req.language or (ag["primary_language"] == req.language or ag["secondary_language"] == req.language)),
            (not req.grp or (req.grp in (ag.get("trained_groups") or []))),
        ])]
    else:
        pool = agents[:]

    # Shift & rest params
    SHIFT_MINUTES = 9 * 60
    REST_MIN = 12 * 60

    # Roster and used set
    roster: List[Dict[str, Any]] = []
    used: set = set()

    # Candidate start sort: by potential gain
    def shift_gain(start_min):
        end_min = start_min + SHIFT_MINUTES
        g = 0
        for t in times:
            if start_min <= t < end_min:
                g += max(demand[t] - assigned[t], 0)
        return g

    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    # Date context (for rest across days)
    day_dt = datetime.fromisoformat(req.date).replace(tzinfo=timezone.utc)  # treat as UTC day

    for start in candidate_starts:
        start_min = start
        end_min = start_min + SHIFT_MINUTES

        # While unmet demand in this span, assign one more agent
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):

            # Pick an unused agent respecting site hours & rest rule
            pick = None
            start_dt_utc = day_dt + timedelta(minutes=start_min)
            for ag in pool:
                if ag["agent_id"] in used:
                    continue

                # Site hours
                if not site_allows(ag.get("site_id"), start_min, end_min):
                    continue

                # Rest across days: last_end + 12h <= start
                last_end = last_end_map.get(ag["agent_id"])
                if last_end and last_end > (start_dt_utc - timedelta(minutes=REST_MIN)):
                    continue

                # Capability (already filtered, but extra safety)
                # If intervals contain multiple langs/grps simultaneously, check peak bucket in span
                ok = False
                for t in times:
                    if start_min <= t < end_min:
                        for iv in times_map[t]:
                            if can_cover(ag, iv["lang"], iv["grp"]):
                                ok = True; break
                    if ok: break
                if not ok:
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

            # increment coverage in the span
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1

            if all(assigned[t] >= demand[t] for t in times):
                break

    # -----------------------------------------------------------------------------
    # Break planning with fairness & capacity subtraction
    # -----------------------------------------------------------------------------
    BREAK_CAP_FRAC = 0.25  # ≤25% of assigned can be on break per interval
    time_list = list(times)  # alias

    def snap_to_grid(m):
        return min(time_list, key=lambda t: abs(t - m)) if time_list else m

    # current break load per interval
    break_load: Dict[int, int] = {t: 0 for t in time_list}

    def cap_at(t):
        base = assigned[t]
        if base <= 0:
            return 0
        return max(1, int(ceil(base * BREAK_CAP_FRAC)))

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
        best = None; best_sc = None
        # pass 1: feasible
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
                    if over > 0:
                        pen += over
                    sc += max(demand[t] - assigned[t], 0)
            if (best_pen is None or pen < best_pen
                or (pen == best_pen and (best_sc is None or sc < best_sc))):
                best_pen = pen; best_sc = sc; best = s
        return best

    def window_candidates(center, duration, lo, hi):
        w_lo = clamp(center - 60, lo, max(lo, hi - duration))
        w_hi = clamp(center + 60, lo, max(lo, hi - duration))
        cands = [t for t in time_list if w_lo <= t <= w_hi]
        return cands or [snap_to_grid(center)]

    for item in roster:
        st_str, en_str = [s.strip() for s in item["shift"].split("-")]
        st_h, st_m = map(int, st_str.split(":")); en_h, en_m = map(int, en_str.split(":"))
        start_min = st_h*60 + st_m; end_min = en_h*60 + en_m
        if end_min <= start_min:
            end_min += 24*60

        # no break in first/last hour; 2h gap between breaks
        NO_HEAD, NO_TAIL, GAP = 60, 60, 120
        place_start = start_min + NO_HEAD
        place_end   = end_min   - NO_TAIL
        if place_end - place_start < (15 + 30 + 15 + 2*GAP):
            item["breaks"] = []; continue

        mid = (start_min + end_min)//2
        lunch_target = clamp(mid, place_start + GAP//2, place_end - GAP//2)
        b1_target = clamp(lunch_target - GAP - 45, place_start, place_end)
        b3_target = clamp(lunch_target + GAP + 45, place_start, place_end)

        # lunch (30)
        lunch_cands = window_candidates(lunch_target, 30, place_start, place_end)
        lunch_s = choose_slot(lunch_cands, 30) or snap_to_grid(lunch_target)
        lunch_e = lunch_s + 30
        for t in time_list:
            if lunch_s <= t < lunch_e: break_load[t] += 1

        # first 15 before lunch
        b1_end_allowed = lunch_s - GAP
        b1_cands = [t for t in window_candidates(b1_target, 15, place_start, place_end)
                    if (t + 15) <= b1_end_allowed]
        if not b1_cands:
            b1_cands = [t for t in time_list if place_start <= t <= max(place_start, lunch_s - GAP - 15)]
        b1_s = choose_slot(b1_cands, 15) if b1_cands else None
        if b1_s is None and place_start + 15 <= b1_end_allowed:
            b1_s = snap_to_grid(max(place_start, min(b1_target, b1_end_allowed - 15)))
        b1_e = b1_s + 15 if b1_s is not None else None
        if b1_s is not None:
            for t in time_list:
                if b1_s <= t < b1_e: break_load[t] += 1

        # last 15 after lunch
        b3_start_allowed = lunch_e + GAP
        b3_cands = [t for t in window_candidates(b3_target, 15, place_start, place_end)
                    if t >= b3_start_allowed]
        if not b3_cands:
            b3_cands = [t for t in time_list if min(place_end-15, b3_target) <= t <= (place_end-15)]
        b3_s = choose_slot(b3_cands, 15) if b3_cands else None
        if b3_s is None and b3_start_allowed <= (place_end-15):
            b3_s = snap_to_grid(min(place_end-15, max(b3_target, b3_start_allowed)))
        b3_e = b3_s + 15 if b3_s is not None else None
        if b3_s is not None:
            for t in time_list:
                if b3_s <= t < b3_e: break_load[t] += 1

        # attach
        item["breaks"] = []
        if b1_s is not None:
            item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)), "end": minutes_to_hhmm(b1_e % (24*60)), "kind": "break15"})
        item["breaks"].append({"start": minutes_to_hhmm(lunch_s % (24*60)), "end": minutes_to_hhmm(lunch_e % (24*60)), "kind": "lunch30"})
        if b3_s is not None:
            item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)), "end": minutes_to_hhmm(b3_e % (24*60)), "kind": "break15"})

    # Subtract break capacity to show actual working capacity
    working: Dict[int, int] = {t: max(0, assigned[t] - break_load[t]) for t in times}

    # -----------------------------------------------------------------------------
    # Persist last_end_utc for used agents (so next day enforces 12h rest)
    # -----------------------------------------------------------------------------
    upserts = []
    for item in roster:
        st_str, en_str = [s.strip() for s in item["shift"].split("-")]
        st_h, st_m = map(int, st_str.split(":")); en_h, en_m = map(int, en_str.split(":"))
        start_min = st_h*60 + st_m; end_min = en_h*60 + en_m
        day0 = day_dt
        # if overnight
        if end_min <= start_min:
            end_dt = (day0 + timedelta(days=1)) + timedelta(minutes=end_min)
        else:
            end_dt = day0 + timedelta(minutes=end_min)
        upserts.append({
            "agent_id": item["agent_id"],
            "last_end_utc": end_dt.isoformat().replace("+00:00", "Z"),
        })

    if upserts:
        try:
            u = httpx.post(
                f"{REST_BASE}/agent_state",
                headers=HEADERS,
                params={"on_conflict": "agent_id"},
                json=upserts,
                timeout=20,
            )
            # If table does not exist, ignore; you can create it with SQL below.
            _ = u.status_code
        except Exception:
            pass

    # Summary payload
    summary = {
        "date": req.date,
        "intervals": [
            {"time": minutes_to_hhmm(t), "req": demand[t], "assigned": assigned[t], "on_break": break_load[t], "working": working[t]}
            for t in times
        ],
        "agents_used": len({r["agent_id"] for r in roster}),
        "roster": roster,
        "notes": [
            "Greedy 9h assignment; site hours enforced; 12h rest across days using agent_state.",
            "Breaks 15/30/15 with fairness cap; 'working' subtracts concurrent breaks from assigned.",
        ],
    }
    return summary
