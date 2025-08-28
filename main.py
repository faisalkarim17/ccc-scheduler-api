from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import os, httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    # still allow root path to work; health will error if missing
    REST_BASE = None
    HEADERS = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

app = FastAPI(title="CCC Scheduler API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # later restrict to your Vercel domain
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
            timeout=10
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

from math import factorial, exp, ceil
from typing import Dict, Any, List

def erlang_c(a: float, N: int) -> float:
    """Erlang C probability of wait (queue non-empty). a = offered load (Erlangs)."""
    if N <= a:  # stability guard
        return 1.0
    summ = 0.0
    for k in range(N):
        summ += (a**k) / factorial(k)
    top = (a**N) / factorial(N) * (N / (N - a))
    return top / (summ + top)

def required_agents_for_target(volume: int, aht_sec: int, interval_seconds: int,
                               target_sl: float, target_t: int) -> int:
    """
    Compute min agents N such that P(wait <= target_t) >= target_sl.
    Uses Erlang C: P(wait > t) = ErlangC * exp(-(N*mu - λ)*t).
    """
    if volume <= 0:
        return 0
    lam = volume / interval_seconds               # arrivals per second
    mu = 1.0 / max(aht_sec, 1)                    # service rate per agent (per second)
    a = lam / mu                                  # offered load (Erlangs)

    N = max(1, ceil(a))                           # start at ceiling of load
    for _ in range(200):                          # simple search cap
        if N <= a:
            N = int(ceil(a)) + 1
        Ec = erlang_c(a, N)
        # P(wait > t)
        p_wait_gt_t = Ec * exp(-(N * mu - lam) * target_t)
        sl = 1.0 - p_wait_gt_t
        if sl >= target_sl:
            return N
        N += 1
    return N

@app.get("/requirements")
def requirements(
    date: str = Query(..., description="YYYY-MM-DD"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    interval_seconds: int = 1800,       # 30 minutes default
    target_sl: float = 0.80,            # 80%
    target_t: int = 20,                 # 20 seconds
    shrinkage: float = 0.30             # 30% default (you can override per call)
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # fetch forecasts for the day (and optional filters)
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
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    rows = r.json()

    # compute requirement per interval bucket (Language × Group × interval_time)
    out: List[Dict[str, Any]] = []
    for row in rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, interval_seconds, target_sl, target_t)
        # apply shrinkage
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

from pydantic import BaseModel
from datetime import datetime, timedelta

class RosterRequest(BaseModel):
    date: str                    # YYYY-MM-DD
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: float = 0.30
    interval_seconds: int = 1800
    target_sl: float = 0.80
    target_t: int = 20

def time_to_minutes(tstr: str) -> int:
    hh, mm, ss = tstr.split(":")
    return int(hh) * 60 + int(mm)  # ignore seconds

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def pick_best_slot(candidate_starts, duration_min, demand_curve, assigned_curve):
    """
    Choose the start minute that minimizes 'stress' during the break window.
    Stress metric = sum(max(demand - assigned, 0)) across the break span.
    """
    best = None
    best_score = None
    for s in candidate_starts:
        e = s + duration_min
        score = 0
        for t in demand_curve:
            if s <= t < e:
                score += max(demand_curve[t] - assigned_curve[t], 0)
        if best_score is None or score < best_score:
            best_score = score
            best = s
    return best

@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # 1) compute requirements for the date (reuse our function/endpoint logic)
    q = {
        "date": req.date,
        "interval_seconds": req.interval_seconds,
        "target_sl": req.target_sl,
        "target_t": req.target_t,
        "shrinkage": req.shrinkage,
    }
    if req.language: q["language"] = req.language
    if req.grp:      q["grp"] = req.grp

    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{req.date}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({ "language": f"eq.{req.language}" } if req.language else {}),
        **({ "grp": f"eq.{req.grp}" } if req.grp else {}),
    }, timeout=30)
    r.raise_for_status()
    fc_rows = r.json()

    # build per-interval requirement using same math:
    intervals = []
    for row in fc_rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, req.interval_seconds, req.target_sl, req.target_t)
        req_after = ceil(N_core / (1.0 - max(0.0, min(req.shrinkage, 0.95)))) if req.shrinkage < 0.99 else N_core
        intervals.append({
            "t": row["interval_time"],
            "lang": row["language"],
            "grp": row["grp"],
            "req": req_after
        })

    # 2) fetch agents
    a = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params={
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": 2000
    }, timeout=30)
    a.raise_for_status()
    agents = a.json()

    # basic filter by language/group capability
    def can_cover(agent, lang, grp):
        lang_ok = (agent["primary_language"] == lang) or (agent["secondary_language"] == lang)
        grp_ok = grp in (agent.get("trained_groups") or [])
        return bool(lang_ok and grp_ok)

    # 3) naive greedy: schedule 9h shift blocks to cover peak first
    # Build interval minutes list for the day
    times = sorted({ time_to_minutes(x["t"]) for x in intervals })
    times_map = { t: [iv for iv in intervals if time_to_minutes(iv["t"])==t] for t in times }

    # demand curve: total required per interval (within filters)
    demand = { t: sum(iv["req"] for iv in times_map[t]) for t in times }
    assigned = { t: 0 for t in times }

    # agents pool filtered by capability for language/grp if specified; else all
    if req.language or req.grp:
        pool = [ag for ag in agents if all([
            (not req.language or (ag["primary_language"]==req.language or ag["secondary_language"]==req.language)),
            (not req.grp or (req.grp in (ag.get("trained_groups") or [])))
        ])]
    else:
        pool = agents[:]

    # simple heuristic: sort intervals by highest unmet demand, assign agents to 9h spans
    SHIFT_MINUTES = 9 * 60
    REST_MIN = 12 * 60

    # track last end time to ensure 12h rest (per single day this is trivial; enforced across days later)
    last_end = {}

    roster = []
    used = set()

    # helper: check if placing a shift [start, start+9h) helps coverage
    def shift_gain(start_min):
        end_min = start_min + SHIFT_MINUTES
        gain = 0
        for t in times:
            if start_min <= t < end_min:
                need = max(demand[t] - assigned[t], 0)
                gain += need
        return gain

    # sort candidate start times on 30-min grid by potential gain (descending)
    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    for start in candidate_starts:
        start_min = start
        end_min = start_min + SHIFT_MINUTES
        # while there is unmet demand in this span, assign one more agent
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
            # pick an unused agent that can work (ignore site hours in stub)
            pick = None
            for ag in pool:
                if ag["agent_id"] in used: 
                    continue
                # rest rule skip (stub; single day has no prior)
                pick = ag
                break
            if not pick:
                break
            # assign this agent
            used.add(pick["agent_id"])
            roster.append({
                "agent_id": pick["agent_id"],
                "full_name": pick["full_name"],
                "date": req.date,
                "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                "notes": "stub assignment"
            })
            # increment coverage in the span
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1

            # stop if all covered
            if all(assigned[t] >= demand[t] for t in times):
                break

  # ... your existing greedy assignment code finishes here and you have `roster`, `times`, `demand`, `assigned` ready

# === Break planning (15/30/15) ===
try:
    time_list = sorted(times)
    def snap_to_grid(m):
        return min(time_list, key=lambda t: abs(t - m)) if time_list else m

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def pick_best_slot(candidate_starts, duration_min, demand_curve, assigned_curve):
        best = None
        best_score = None
        for s in candidate_starts:
            e = s + duration_min
            score = 0
            for t in demand_curve:
                if s <= t < e:
                    score += max(demand_curve[t] - assigned_curve[t], 0)
            if best_score is None or score < best_score:
                best_score = score
                best = s
        return best

    for item in roster:
        st_str, en_str = [s.strip() for s in item["shift"].split("-")]
        st_h, st_m = map(int, st_str.split(":"))
        en_h, en_m = map(int, en_str.split(":"))
        start_min = st_h*60 + st_m
        end_min = en_h*60 + en_m
        if end_min <= start_min:
            end_min += 24*60

        NO_HEAD, NO_TAIL, GAP = 60, 60, 120
        place_start = start_min + NO_HEAD
        place_end   = end_min   - NO_TAIL
        if place_end - place_start < (15+30+15 + 2*GAP):
            item["breaks"] = []
            continue

        mid = (start_min + end_min)//2
        lunch_target = clamp(mid, place_start+GAP//2, place_end-GAP//2)
        b1_target = clamp(lunch_target - GAP - 45, place_start, place_end)
        b3_target = clamp(lunch_target + GAP + 45, place_start, place_end)

        def window(center, dur):
            w_lo = clamp(center - 60, place_start, place_end - dur)
            w_hi = clamp(center + 60, place_start, place_end - dur)
            c = [t for t in time_list if w_lo <= t <= w_hi]
            return c or [snap_to_grid(center)]

        lunch_c = window(lunch_target, 30)
        lunch_s = pick_best_slot(lunch_c, 30, demand, assigned) or snap_to_grid(lunch_target)
        lunch_e = lunch_s + 30

        b1_end_allowed = lunch_s - GAP
        b1_c = [t for t in window(b1_target, 15) if (t+15) <= b1_end_allowed]
        if not b1_c:
            b1_c = [t for t in time_list if place_start <= t <= max(place_start, lunch_s - GAP - 15)]
        b1_s = pick_best_slot(b1_c, 15, demand, assigned) if b1_c else None
        if b1_s is None and place_start + 15 <= b1_end_allowed:
            b1_s = snap_to_grid(max(place_start, min(b1_target, b1_end_allowed - 15)))
        b1_e = b1_s + 15 if b1_s is not None else None

        b3_start_allowed = lunch_e + GAP
        b3_c = [t for t in window(b3_target, 15) if t >= b3_start_allowed]
        if not b3_c:
            b3_c = [t for t in time_list if min(place_end-15, b3_target) <= t <= (place_end-15)]
        b3_s = pick_best_slot(b3_c, 15, demand, assigned) if b3_c else None
        if b3_s is None and b3_start_allowed <= (place_end-15):
            b3_s = snap_to_grid(min(place_end-15, max(b3_target, b3_start_allowed)))
        b3_e = b3_s + 15 if b3_s is not None else None

        item["breaks"] = []
        if b1_s is not None:
            item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)),
                                   "end": minutes_to_hhmm(b1_e % (24*60)), "kind": "break15"})
        item["breaks"].append({"start": minutes_to_hhmm(lunch_s % (24*60)),
                               "end": minutes_to_hhmm(lunch_e % (24*60)), "kind": "lunch30"})
        if b3_s is not None:
            item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)),
                                   "end": minutes_to_hhmm(b3_e % (24*60)), "kind": "break15"})
except Exception as e:
    print("[break planning] skipped due to error:", e)
    for item in roster:
        item.setdefault("breaks", [])

# NOW build summary (after breaks are attached)
summary = {
    "date": req.date,
    "intervals": [{"time": minutes_to_hhmm(t), "req": demand[t], "assigned": assigned[t]} for t in times],
    "agents_used": len(used),
    "roster": roster,
    "notes": ["MVP stub: same-day 9h greedy coverage with 15/30/15 breaks. Next: subtract breaks from capacity, fairness, rest rules."]
}
return summary
