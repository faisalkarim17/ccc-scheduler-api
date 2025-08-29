# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx

# ---------- Supabase REST setup ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

# ---------- FastAPI app ----------
app = FastAPI(title="CCC Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def erlang_c(a: float, N: int) -> float:
    """Erlang C probability of wait (queue non-empty). a = offered load (Erlangs)."""
    if N <= a:
        return 1.0
    s = 0.0
    for k in range(N):
        s += (a**k) / factorial(k)
    top = (a**N) / factorial(N) * (N / (N - a))
    return top / (s + top)

def required_agents_for_target(volume: int, aht_sec: int, interval_seconds: int,
                               target_sl: float, target_t: int) -> int:
    """Min N s.t. P(wait<=target_t) >= target_sl, via Erlang C."""
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

def time_to_minutes(tstr: str) -> int:
    hh, mm, ss = tstr.split(":")
    return int(hh) * 60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------- Models ----------
class RosterRequest(BaseModel):
    date: str                    # YYYY-MM-DD
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: float = 0.30
    interval_seconds: int = 1800
    target_sl: float = 0.80
    target_t: int = 20

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

@app.get("/requirements")
def requirements(
    date: str = Query(..., description="YYYY-MM-DD"),
    language: Optional[str] = None,
    grp: Optional[str] = None,
    interval_seconds: int = 1800,
    target_sl: float = 0.80,
    target_t: int = 20,
    shrinkage: float = 0.30
):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

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

# ---------- Generate roster with break fairness ----------
@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # 1) pull the day's forecasts (optionally filtered)
    r = httpx.get(
        f"{REST_BASE}/forecasts",
        headers=HEADERS,
        params={
            "select": "date,interval_time,language,grp,service,volume,aht_sec",
            "date": f"eq.{req.date}",
            "order": "interval_time.asc,language.asc,grp.asc",
            **({"language": f"eq.{req.language}"} if req.language else {}),
            **({"grp": f"eq.{req.grp}"} if req.grp else {}),
        },
        timeout=30
    )
    r.raise_for_status()
    fc_rows = r.json()
    if not fc_rows:
        raise HTTPException(400, "No forecast rows for that date/filters")

    # compute required staff per interval
    intervals: List[Dict[str, Any]] = []
    for row in fc_rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, req.interval_seconds, req.target_sl, req.target_t)
        req_after = N_core if req.shrinkage >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(req.shrinkage, 0.95))))
        intervals.append({"t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after})

    # 2) agents
    a = httpx.get(
        f"{REST_BASE}/agents",
        headers=HEADERS,
        params={"select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
                "limit": 2000},
        timeout=30
    )
    a.raise_for_status()
    agents = a.json()

    # capability filter
    if req.language or req.grp:
        pool = [
            ag for ag in agents
            if (not req.language or ag["primary_language"] == req.language or ag["secondary_language"] == req.language)
            and (not req.grp or req.grp in (ag.get("trained_groups") or []))
        ]
    else:
        pool = agents[:]

    # 3) greedy 9h shifts to cover demand
    times = sorted({time_to_minutes(x["t"]) for x in intervals})
    times_map = {t: [iv for iv in intervals if time_to_minutes(iv["t"]) == t] for t in times}
    demand = {t: sum(iv["req"] for iv in times_map[t]) for t in times}
    assigned = {t: 0 for t in times}

    SHIFT_MINUTES = 9 * 60
    roster: List[Dict[str, Any]] = []
    used = set()

    def shift_gain(start_min: int) -> int:
        end_min = start_min + SHIFT_MINUTES
        gain = 0
        for t in times:
            if start_min <= t < end_min:
                gain += max(demand[t] - assigned[t], 0)
        return gain

    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    for start in candidate_starts:
        start_min = start
        end_min = start_min + SHIFT_MINUTES
        # keep adding agents as long as that span is under-staffed
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
            pick = None
            for ag in pool:
                if ag["agent_id"] in used:
                    continue
                pick = ag
                break
            if not pick:
                break
            used.add(pick["agent_id"])
            roster.append({
                "agent_id": pick["agent_id"],
                "full_name": pick["full_name"],
                "date": req.date,
                "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                "notes": "stub assignment"
            })
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1
            if all(assigned[t] >= demand[t] for t in times):
                break

    # 4) Break planning + fairness/staggering (15/30/15)
    try:
        time_list = sorted(times)

        def snap_to_grid(m):
            return min(time_list, key=lambda t: abs(t - m)) if time_list else m

        # fairness cap: at most 25% of assigned agents per interval on break (>=1 if any working)
        BREAK_CAP_FRAC = 0.25
        break_load = {t: 0 for t in time_list}

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
            best = None
            best_sc = None
            # pass 1: feasible under cap, minimize stress
            for s in cands:
                if span_ok(s, dur):
                    sc = stress(s, dur)
                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best = s
            if best is not None:
                return best
            # pass 2: minimal cap violation, then least stress
            best = None
            best_pen = None
            best_sc = None
            for s in cands:
                e = s + dur
                pen = 0
                sc = 0
                for t in time_list:
                    if s <= t < e:
                        over = (break_load[t] + 1) - cap_at(t)
                        if over > 0:
                            pen += over
                        sc += max(demand[t] - assigned[t], 0)
                if (best_pen is None or pen < best_pen
                        or (pen == best_pen and (best_sc is None or sc < best_sc))):
                    best_pen = pen
                    best_sc = sc
                    best = s
            return best

        def window_candidates(center, duration, lo, hi):
            w_lo = clamp(center - 60, lo, max(lo, hi - duration))
            w_hi = clamp(center + 60, lo, max(lo, hi - duration))
            cands = [t for t in time_list if w_lo <= t <= w_hi]
            return cands or [snap_to_grid(center)]

        for item in roster:
            st_str, en_str = [s.strip() for s in item["shift"].split("-")]
            st_h, st_m = map(int, st_str.split(":"))
            en_h, en_m = map(int, en_str.split(":"))
            start_min = st_h * 60 + st_m
            end_min = en_h * 60 + en_m
            if end_min <= start_min:
                end_min += 24 * 60  # overnight

            NO_HEAD, NO_TAIL, GAP = 60, 60, 120
            place_start = start_min + NO_HEAD
            place_end = end_min - NO_TAIL
            if place_end - place_start < (15 + 30 + 15 + 2 * GAP):
                item["breaks"] = []
                continue

            mid = (start_min + end_min) // 2
            lunch_target = clamp(mid, place_start + GAP // 2, place_end - GAP // 2)
            b1_target = clamp(lunch_target - GAP - 45, place_start, place_end)
            b3_target = clamp(lunch_target + GAP + 45, place_start, place_end)

            # lunch first (30)
            lunch_cands = window_candidates(lunch_target, 30, place_start, place_end)
            lunch_s = choose_slot(lunch_cands, 30) or snap_to_grid(lunch_target)
            lunch_e = lunch_s + 30
            for t in time_list:
                if lunch_s <= t < lunch_e:
                    break_load[t] += 1

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
                    if b1_s <= t < b1_e:
                        break_load[t] += 1

            # last 15 after lunch
            b3_start_allowed = lunch_e + GAP
            b3_cands = [t for t in window_candidates(b3_target, 15, place_start, place_end)
                        if t >= b3_start_allowed]
            if not b3_cands:
                b3_cands = [t for t in time_list if min(place_end - 15, b3_target) <= t <= (place_end - 15)]
            b3_s = choose_slot(b3_cands, 15) if b3_cands else None
            if b3_s is None and b3_start_allowed <= (place_end - 15):
                b3_s = snap_to_grid(min(place_end - 15, max(b3_target, b3_start_allowed)))
            b3_e = b3_s + 15 if b3_s is not None else None
            if b3_s is not None:
                for t in time_list:
                    if b3_s <= t < b3_e:
                        break_load[t] += 1

            item["breaks"] = []
            if b1_s is not None:
                item["breaks"].append({
                    "start": minutes_to_hhmm(b1_s % (24 * 60)),
                    "end": minutes_to_hhmm(b1_e % (24 * 60)),
                    "kind": "break15"
                })
            item["breaks"].append({
                "start": minutes_to_hhmm(lunch_s % (24 * 60)),
                "end": minutes_to_hhmm(lunch_e % (24 * 60)),
                "kind": "lunch30"
            })
            if b3_s is not None:
                item["breaks"].append({
                    "start": minutes_to_hhmm(b3_s % (24 * 60)),
                    "end": minutes_to_hhmm(b3_e % (24 * 60)),
                    "kind": "break15"
                })

    except Exception as e:
        print("[break planning] skipped due to error:", e)
        for item in roster:
            item.setdefault("breaks", [])

    # 5) summary
    summary = {
        "date": req.date,
        "intervals": [{"time": minutes_to_hhmm(t), "req": demand[t], "assigned": assigned[t]} for t in times],
        "agents_used": len(used),
        "roster": roster,
        "notes": [
            "MVP stub: 9h greedy coverage with fairness-staggered 15/30/15 breaks.",
            "Next: subtract break capacity from assigned, add fairness balancing & rest windows across days."
        ],
    }
    return summary
