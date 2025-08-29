from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from math import factorial, exp, ceil
import os, httpx, json

# ---------- Supabase env ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    REST_BASE = None
    HEADERS = {}
else:
    REST_BASE = f"{SUPABASE_URL}/rest/v1"
    HEADERS = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"}

# ---------- App ----------
app = FastAPI(title="CCC Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten later to your front-end origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config (in-memory, editable via /config) ----------
CONFIG: Dict[str, Any] = {
    "interval_seconds": 1800,
    "target_sl": 0.80,
    "target_t": 20,
    "shrinkage": 0.30,

    # scheduling knobs
    "shift_minutes": 9 * 60,           # default 9h
    "break_pattern": [15, 30, 15],     # minutes, in order
    "break_cap_frac": 0.25,            # <=25% of assigned on break per interval

    # placement rules
    "no_head": 60,                     # no breaks first X minutes
    "no_tail": 60,                     # no breaks last X minutes
    "lunch_gap": 120,                  # min gap b/w 15-30,30-15

    # site hours + cross-day rest
    "site_hours_enforced": False,
    "site_hours": {},                  # { "QA": {"open": "10:00", "close": "19:00"}, ... }
    "rest_min_minutes": 12 * 60,       # 12h
    "prev_end_times": {},              # {"A001":"2025-09-06T19:00:00", ...}

    "timezone": "UTC",
}

# ---------- Helpers ----------
def time_to_minutes_hhmm(tstr: str) -> int:
    # tstr "HH:MM" or "HH:MM:SS"
    parts = tstr.split(":")
    hh, mm = int(parts[0]), int(parts[1])
    return hh * 60 + mm

def time_to_minutes(interval_time: str) -> int:
    # from forecasts interval_time "HH:MM:SS"
    hh, mm, ss = interval_time.split(":")
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

# ---------- Models ----------
class RosterRequest(BaseModel):
    date: str
    language: Optional[str] = None
    grp: Optional[str] = None
    shrinkage: Optional[float] = None
    interval_seconds: Optional[int] = None
    target_sl: Optional[float] = None
    target_t: Optional[int] = None

class RosterRangeRequest(BaseModel):
    date_from: str   # YYYY-MM-DD
    date_to: str     # YYYY-MM-DD
    language: Optional[str] = None
    grp: Optional[str] = None

# ---------- Routes: basics ----------
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

# ---------- Config endpoints ----------
@app.get("/config")
def get_config():
    return CONFIG

@app.put("/config")
def put_config(body: Dict[str, Any]):
    CONFIG.update(body or {})
    return CONFIG

# ---------- Data fetchers ----------
@app.get("/agents")
def list_agents(limit: int = 2000, offset: int = 0, site_id: Optional[str] = None):
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
        "order": "interval_time.asc,language.asc,grp.asc",
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

    isec = interval_seconds or CONFIG["interval_seconds"]
    tsl  = target_sl or CONFIG["target_sl"]
    tt   = target_t or CONFIG["target_t"]
    shr  = CONFIG["shrinkage"] if shrinkage is None else shrinkage

    params = {
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{date}",
        "order": "interval_time.asc,language.asc,grp.asc",
    }
    if language: params["language"] = f"eq.{language}"
    if grp:      params["grp"] = f"eq.{grp}"

    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    rows = r.json()

    out: List[Dict[str, Any]] = []
    for row in rows:
        vol = int(row["volume"])
        aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, isec, tsl, tt)
        req = N_core if shr >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(shr, 0.95))))
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

# ---------- Core roster generator (single day) ----------
def generate_roster_core(req: RosterRequest, prev_end_times: Dict[str, str]) -> Dict[str, Any]:
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")

    # pull config
    isec    = req.interval_seconds or CONFIG["interval_seconds"]
    tsl     = req.target_sl or CONFIG["target_sl"]
    tt      = req.target_t or CONFIG["target_t"]
    shr     = CONFIG["shrinkage"] if req.shrinkage is None else req.shrinkage
    SHIFT   = CONFIG["shift_minutes"]
    NO_HEAD = CONFIG["no_head"]; NO_TAIL = CONFIG["no_tail"]; GAP = CONFIG["lunch_gap"]
    BREAKS  = CONFIG["break_pattern"]        # e.g., [15,30,15]
    CAP_FR  = CONFIG["break_cap_frac"]
    SITE_ON = CONFIG["site_hours_enforced"]
    SITE_H  = CONFIG["site_hours"]
    RESTMIN = CONFIG["rest_min_minutes"]

    # 1) fetch forecasts
    r = httpx.get(f"{REST_BASE}/forecasts", headers=HEADERS, params={
        "select": "date,interval_time,language,grp,service,volume,aht_sec",
        "date": f"eq.{req.date}",
        "order": "interval_time.asc,language.asc,grp.asc",
        **({"language": f"eq.{req.language}"} if req.language else {}),
        **({"grp": f"eq.{req.grp}"} if req.grp else {}),
    }, timeout=30)
    r.raise_for_status()
    fc_rows = r.json()

    # requirement per interval
    intervals = []
    for row in fc_rows:
        vol = int(row["volume"]); aht = int(row["aht_sec"])
        N_core = required_agents_for_target(vol, aht, isec, tsl, tt)
        req_after = N_core if shr >= 0.99 else ceil(N_core / (1.0 - max(0.0, min(shr, 0.95))))
        intervals.append({
            "t": row["interval_time"], "lang": row["language"], "grp": row["grp"], "req": req_after
        })

    # 2) fetch agents
    a = httpx.get(f"{REST_BASE}/agents", headers=HEADERS, params={
        "select": "agent_id,full_name,site_id,primary_language,secondary_language,trained_groups,trained_services",
        "limit": 2000
    }, timeout=30)
    a.raise_for_status()
    agents = a.json()

    # filters
    def agent_ok(ag, lang, grp):
        lang_ok = (ag["primary_language"]==lang) or (ag["secondary_language"]==lang)
        grp_ok  = grp in (ag.get("trained_groups") or [])
        return bool(lang_ok and grp_ok)

    if req.language or req.grp:
        pool = [ag for ag in agents if agent_ok(ag, req.language or ag["primary_language"], req.grp or "")]
    else:
        pool = agents[:]

    # time grid
    times = sorted({ time_to_minutes(x["t"]) for x in intervals })
    times_map = { t: [iv for iv in intervals if time_to_minutes(iv["t"])==t] for t in times }
    demand = { t: sum(iv["req"] for iv in times_map[t]) for t in times }
    assigned = { t: 0 for t in times }

    # helper site-window check
    def within_site(ag_site: Optional[str], start_min: int, end_min: int) -> bool:
        if not SITE_ON: return True
        if not ag_site: return True
        sh = SITE_H.get(ag_site)
        if not sh: return True
        try:
            o = time_to_minutes_hhmm(sh["open"])
            c = time_to_minutes_hhmm(sh["close"])
        except Exception:
            return True
        return (o <= start_min) and (end_min <= c)

    # cross-day rest check
    def rest_ok(agent_id: str, start_min: int, date_str: str) -> bool:
        # if no prev end, ok
        last_end = prev_end_times.get(agent_id)
        if not last_end:
            return True
        try:
            prev_dt = datetime.fromisoformat(last_end)
        except Exception:
            return True
        # today's start as datetime in UTC (date + start_min)
        today = datetime.fromisoformat(date_str + "T00:00:00")
        start_dt = today + timedelta(minutes=start_min)
        return (start_dt - prev_dt).total_seconds() / 60.0 >= RESTMIN

    # greedy: cover unmet demand with SHIFT spans
    roster = []
    used_today = set()

    def shift_gain(start_min):
        end_min = start_min + SHIFT
        g = 0
        for t in times:
            if start_min <= t < end_min:
                g += max(demand[t] - assigned[t], 0)
        return g

    candidate_starts = sorted({t for t in times})
    candidate_starts.sort(key=lambda s: -shift_gain(s))

    for start in candidate_starts:
        start_min = start
        end_min = start_min + SHIFT
        while any(start_min <= t < end_min and assigned[t] < demand[t] for t in times):
            pick = None
            for ag in pool:
                if ag["agent_id"] in used_today:
                    continue
                # site window + rest
                if not within_site(ag.get("site_id"), start_min, end_min):
                    continue
                if not rest_ok(ag["agent_id"], start_min, req.date):
                    continue
                pick = ag
                break
            if not pick: break
            used_today.add(pick["agent_id"])
            roster.append({
                "agent_id": pick["agent_id"],
                "full_name": pick["full_name"],
                "site_id": pick.get("site_id"),
                "date": req.date,
                "shift": f"{minutes_to_hhmm(start_min)} - {minutes_to_hhmm(end_min % (24*60))}",
                "notes": "stub assignment",
                "breaks": []  # filled below
            })
            for t in times:
                if start_min <= t < end_min:
                    assigned[t] += 1
            if all(assigned[t] >= demand[t] for t in times):
                break

    # === Break planning (forecast-driven grid) with fairness cap ===
    try:
        time_list = sorted(times)
        def snap_to_grid(m): return min(time_list, key=lambda t: abs(t - m)) if time_list else m
        from math import ceil as _ceil

        break_load = {t: 0 for t in time_list}
        def cap_at(t):
            base = assigned[t]
            if base <= 0: return 0
            return max(1, int(_ceil(base * CAP_FR)))

        def span_ok(s, dur):
            e = s + dur
            for t in time_list:
                if s <= t < e:
                    if break_load[t] + 1 > cap_at(t): return False
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
                        best_sc, best = sc, s
            if best is not None: return best
            # minimal cap violation if nothing feasible
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
                    best_pen, best_sc, best = pen, sc, s
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
            if end_min <= start_min: end_min += 24*60

            place_start = start_min + NO_HEAD
            place_end   = end_min   - NO_TAIL
            if place_end - place_start < sum(BREAKS) + 2*GAP:
                item["breaks"] = []; continue

            mid = (start_min + end_min)//2
            # simple pattern [15,30,15]
            b15a, b30, b15b = BREAKS
            lunch_target = clamp(mid, place_start + GAP//2, place_end - GAP//2)
            b1_target = clamp(lunch_target - GAP - 45, place_start, place_end)
            b3_target = clamp(lunch_target + GAP + 45, place_start, place_end)

            # lunch 30
            lunch_cands = window_candidates(lunch_target, b30, place_start, place_end)
            l_s = choose_slot(lunch_cands, b30) or snap_to_grid(lunch_target)
            l_e = l_s + b30
            for t in time_list:
                if l_s <= t < l_e: break_load[t] += 1

            # 1st 15 before lunch
            b1_end_allowed = l_s - GAP
            b1_cands = [t for t in window_candidates(b1_target, b15a, place_start, place_end) if (t + b15a) <= b1_end_allowed]
            if not b1_cands:
                b1_cands = [t for t in time_list if place_start <= t <= max(place_start, l_s - GAP - b15a)]
            b1_s = choose_slot(b1_cands, b15a) if b1_cands else None
            if b1_s is None and place_start + b15a <= b1_end_allowed:
                b1_s = snap_to_grid(max(place_start, min(b1_target, b1_end_allowed - b15a)))
            b1_e = b1_s + b15a if b1_s is not None else None
            if b1_s is not None:
                for t in time_list:
                    if b1_s <= t < b1_e: break_load[t] += 1

            # 2nd 15 after lunch
            b3_start_allowed = l_e + GAP
            b3_cands = [t for t in window_candidates(b3_target, b15b, place_start, place_end) if t >= b3_start_allowed]
            if not b3_cands:
                b3_cands = [t for t in time_list if min(place_end-b15b, b3_target) <= t <= (place_end-b15b)]
            b3_s = choose_slot(b3_cands, b15b) if b3_cands else None
            if b3_s is None and b3_start_allowed <= (place_end-b15b):
                b3_s = snap_to_grid(min(place_end-b15b, max(b3_target, b3_start_allowed)))
            b3_e = b3_s + b15b if b3_s is not None else None
            if b3_s is not None:
                for t in time_list:
                    if b3_s <= t < b3_e: break_load[t] += 1

            item["breaks"] = []
            if b1_s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(b1_s % (24*60)),
                                       "end": minutes_to_hhmm(b1_e % (24*60)), "kind": "break15"})
            item["breaks"].append({"start": minutes_to_hhmm(l_s % (24*60)),
                                   "end": minutes_to_hhmm(l_e % (24*60)), "kind": "lunch30"})
            if b3_s is not None:
                item["breaks"].append({"start": minutes_to_hhmm(b3_s % (24*60)),
                                       "end": minutes_to_hhmm(b3_e % (24*60)), "kind": "break15"})
    except Exception as e:
        print("[break planning] skipped due to error:", e)
        for item in roster:
            item.setdefault("breaks", [])

    # ---------- subtract breaks from coverage in summary ----------
    def overlaps_interval(bstart: int, bend: int, t: int, isec: int) -> bool:
        # count if any overlap with [t, t+isec)
        return not (bend <= t or (t + isec) <= bstart)

    on_break = {t: 0 for t in times}
    for item in roster:
        for b in item.get("breaks", []):
            bs = time_to_minutes_hhmm(b["start"])
            be = time_to_minutes_hhmm(b["end"])
            if be <= bs: be += 24*60
            for t in times:
                if overlaps_interval(bs, be, t, isec):
                    on_break[t] += 1

    summary_intervals = []
    for t in times:
        work = max(0, assigned[t] - on_break[t])
        summary_intervals.append({
            "time": minutes_to_hhmm(t),
            "req":  demand[t],
            "assigned": assigned[t],
            "on_break": on_break[t],
            "working": work
        })

    return {
        "date": req.date,
        "intervals": summary_intervals,
        "agents_used": len(set(x["agent_id"] for x in roster)),
        "roster": roster,
        "notes": [
            "Greedy 9h assignment; fairness cap; site hours enforced (if enabled); "
            "12h cross-day rest via prev_end_times.",
            "Config-driven: shift_minutes, break_pattern, caps, targets, shrinkage, site_hours, rest_min_minutes."
        ]
    }

# ---------- POST /generate-roster (single day) ----------
@app.post("/generate-roster")
def generate_roster(req: RosterRequest):
    # copy prev_end map so we don't mutate CONFIG while generating one day
    prev_map = dict(CONFIG.get("prev_end_times", {}))
    out = generate_roster_core(req, prev_map)
    return out

# ---------- POST /generate-roster-range (multi day) ----------
@app.post("/generate-roster-range")
def generate_roster_range(body: RosterRangeRequest):
    # walk from date_from to date_to, carrying prev_end_times
    try:
        d0 = datetime.fromisoformat(body.date_from).date()
        d1 = datetime.fromisoformat(body.date_to).date()
    except Exception:
        raise HTTPException(400, "date_from/date_to must be YYYY-MM-DD")

    if d1 < d0:
        raise HTTPException(400, "date_to must be >= date_from")

    prev_map = dict(CONFIG.get("prev_end_times", {}))
    results: Dict[str, Any] = {}
    cur = d0
    while cur <= d1:
        req = RosterRequest(
            date=str(cur),
            language=body.language,
            grp=body.grp
        )
        res = generate_roster_core(req, prev_map)
        results[str(cur)] = res
        # update prev_end_times for next day from this day's roster
        for item in res["roster"]:
            sh = item["shift"]; st_str, en_str = [s.strip() for s in sh.split("-")]
            # assume shift end is same date
            end_dt = datetime.fromisoformat(f"{cur}T{en_str}")
            prev_map[item["agent_id"]] = end_dt.isoformat()
        cur += timedelta(days=1)
    return results

# ---------- (Optional) save to Supabase ----------
# Requires a table public.rosters with columns:
# date (date), agent_id (text), full_name (text), site_id (text),
# start_time (time), end_time (time), breaks (jsonb), meta (jsonb)
class SaveRosterPayload(BaseModel):
    date: str
    roster: List[Dict[str, Any]]

@app.post("/save-roster")
def save_roster(payload: SaveRosterPayload):
    if not REST_BASE:
        raise HTTPException(500, "Supabase env vars missing")
    rows = []
    for r in payload.roster:
        st_str, en_str = [s.strip() for s in r["shift"].split("-")]
        rows.append({
            "date": payload.date,
            "agent_id": r["agent_id"],
            "full_name": r.get("full_name"),
            "site_id": r.get("site_id"),
            "start_time": st_str + ":00",
            "end_time":   en_str + ":00",
            "breaks": r.get("breaks", []),
            "meta": {"notes": r.get("notes")}
        })
    # upsert by (date, agent_id)
    resp = httpx.post(
        f"{REST_BASE}/rosters",
        headers={**HEADERS, "Prefer": "resolution=merge-duplicates"},
        json=rows, timeout=30
    )
    if resp.status_code not in (200, 201, 204):
        # table might not exist yet – return clean error
        raise HTTPException(resp.status_code, resp.text)
    return {"status": "ok", "inserted": len(rows)}

# ---------- Playground (HTML) ----------
PLAYGROUND_HTML = """<!doctype html><html><head><meta charset="utf-8"/>
<title>CCC Scheduler Playground</title>
<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1024px;margin:24px auto;padding:0 16px}
label{display:block;margin:8px 0 4px;color:#333}input{padding:8px;font-size:14px;width:100%}
button{padding:8px 12px;margin-right:8px;background:#0b6bff;color:#fff;border-radius:8px;border:none}
pre{background:#f5f7fb;border:1px solid #e4e9f2;padding:12px;border-radius:8px;overflow:auto}
.row{display:flex;gap:12px;flex-wrap:wrap}
.row>div{flex:1 min(240px,100%)} .hl{opacity:.6;font-size:12px;margin-top:8px}
</style></head><body>
<h1>CCC Scheduler — Playground</h1>
<p>Quickly try <code>/requirements</code>, <code>/generate-roster</code> and <code>/generate-roster-range</code></p>
<div class="row">
  <div><label>Date</label><input id="date" value="2025-09-07"></div>
  <div><label>Language</label><input id="lang" value="EN"></div>
  <div><label>Group</label><input id="grp" value="G1"></div>
  <div><label>Shrinkage</label><input id="shr" value="0.30"></div>
  <div><label>Target SL</label><input id="tsl" value="0.8"></div>
  <div><label>Target T (sec)</label><input id="tt" value="20"></div>
</div>
<div class="row">
  <div><label>Range: From</label><input id="dfrom" value="2025-09-07"></div>
  <div><label>Range: To</label><input id="dto" value="2025-09-08"></div>
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
function jv(v){ out.textContent = JSON.stringify(v,null,2); }
function vals(){
  return {
    date: document.getElementById('date').value,
    language: document.getElementById('lang').value,
    grp: document.getElementById('grp').value,
    shrinkage: parseFloat(document.getElementById('shr').value),
    target_sl: parseFloat(document.getElementById('tsl').value),
    target_t: parseInt(document.getElementById('tt').value,10)
  };
}
function runReq(){
  const v = vals();
  const q = new URLSearchParams({
    date: v.date, language: v.language, grp: v.grp,
    target_sl: v.target_sl, target_t: v.target_t
  });
  fetch('/requirements?'+q).then(r=>r.json()).then(jv).catch(e=>jv({error:String(e)}));
}
function runRoster(){
  const v = vals();
  fetch('/generate-roster',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)})
    .then(r=>r.json()).then(jv).catch(e=>jv({error:String(e)}));
}
function runRange(){
  const body = {
    date_from: document.getElementById('dfrom').value,
    date_to: document.getElementById('dto').value,
    language: document.getElementById('lang').value,
    grp: document.getElementById('grp').value
  };
  fetch('/generate-roster-range',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(jv).catch(e=>jv({error:String(e)}));
}
</script></body></html>"""

@app.get("/playground")
def playground():
    return HTMLResponse(PLAYGROUND_HTML)
