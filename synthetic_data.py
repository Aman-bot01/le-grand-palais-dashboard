"""
Synthetic data generator for the "safe clone" of the game-analytics dashboard.

Produces C:\\AI\\clone\\synthetic.duckdb containing 100% fabricated data shaped like
the schema launch.py / launch_dashboard_v2.py expect (columns were read directly out
of the real SQL query text in those two files, NOT guessed). No real Pong Studios
game name, location name, or figure appears anywhere in this file.

Fictional brand: "Le Grand Palais" -- a physical casino chain spread across Québec
(flagship market), Ontario (secondary market), and a curated spread of US states
(expansion markets). Revenue intensity is tuned per-city so that hierarchy actually
shows up in the generated numbers, not just in labels.

Tables built:
  GameCatalogView1               - fake game catalog (v1 / v2 / igaming)
  AnalyticsGameTerminalsGames    - daily per-terminal-location rows for v1/v2 games
  TaskHandlerBetSpinSummary      - trailing-window activity feed for v1/v2 (vendor test rig)
  BetSpinSummaryCashView3        - base igaming (PFH + EdgeLabs) spin/bet fact table
  BetSpinSummaryCashView3Pong    - VIEW: PlatformName='Pong' AND CasinoName='PFH' slice
  BetSpinSummaryCashView3EdgeLabs- VIEW: PlatformName='EdgeLabs' slice
  BetSpinSummarySocialView2      - VIEW: same EdgeLabs slice, used by the Social Casino tab
  CrmLocationView                - fake locations (Le Grand Palais properties)
  CrmUpdateLogView               - fake release / math-update log
  LocationAnalyticsSummary       - fake weekly location performance (Math Impact tab)

Run:  python synthetic_data.py
"""
from __future__ import annotations

import os
import datetime as dt

import numpy as np
import pandas as pd
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "synthetic.duckdb")

SEED = 42
rng = np.random.default_rng(SEED)

TODAY = dt.date(2026, 9, 2)  # fixed "as of" so the dataset is reproducible

# ─────────────────────────────────────────────────────────────────────────
# Fake vocabularies -- nothing below is a real Pong Studios asset
# ─────────────────────────────────────────────────────────────────────────
ADJ = ["Cosmic", "Neon", "Lucky", "Golden", "Wild", "Mystic", "Blazing", "Frozen",
       "Crimson", "Sapphire", "Electric", "Diamond", "Silver", "Emerald", "Thunder",
       "Phantom", "Royal", "Savage", "Radiant", "Turbo", "Ancient", "Solar", "Lunar",
       "Rogue", "Velvet", "Iron", "Jade", "Scarlet", "Obsidian", "Prism", "Molten",
       "Frosty", "Wicked", "Sunny", "Midnight", "Feral", "Gilded", "Stormy", "Rustic"]
NOUN = ["Reels", "Falcon", "Vault", "Fortune", "Serpent", "Dragon", "Bandit",
        "Pharaoh", "Tiger", "Wolf", "Phoenix", "Buffalo", "Pirate", "Jackpot",
        "Star", "Comet", "Storm", "Griffin", "Cobra", "Panther", "Rhino", "Eagle",
        "Viking", "Samurai", "Wizard", "Genie", "Kraken", "Mermaid", "Outlaw",
        "Gladiator", "Scarab", "Totem", "Cascade", "Nomad", "Oracle", "Rocket"]
SUFFIX = ["", "", "", "", " 2", " 3", " Deluxe", " Gold", " XL", " Plus", " Extreme"]

CASINO_ADJ = ["Neptune", "Golden", "Silver", "Emerald", "Ruby", "Diamond", "Royal",
              "Grand", "Sunset", "Starlight", "Crown", "Majestic", "Platinum",
              "Coastal", "Highland", "Copper", "Velvet", "Aurora", "Frontier", "Summit"]
CASINO_NOUN = ["Palace", "Bay", "Resort", "Club", "Lounge", "Casino", "Vegas",
               "Cove", "Peak", "Grove"]

STUDIOS = ["Nova Studio", "Aurora Labs", "Pixel Foundry", "Quantum Play",
           "Starforge Games", "Lucky Byte Studios", "Bright Spark Interactive"]
DISTRIBUTORS = ["Apex Gaming Distribution", "Northstar Route", "BluePeak Supply Co",
                "Redwood Gaming Partners", "Frontier Amusement Group", "Copperline Distributors"]
OPERATORS = ["Bright Leaf Gaming", "Ironwood Amusements", "Blue Harbor Entertainment",
             "Timberline Gaming Group", "Coastal Route Amusements"]
FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
               "Drew", "Sam", "Cameron", "Reese", "Avery", "Quinn", "Skyler"]
LAST_NAMES = ["Bennett", "Carver", "Donovan", "Ellison", "Fenwick", "Grayson",
              "Halloway", "Ingram", "Jasper", "Kingsley", "Lockwood", "Marsh"]
# Québec-flavoured names for that region's account managers -- a small authenticity
# touch, not load-bearing anywhere.
FR_FIRST_NAMES = ["Émile", "Camille", "Léa", "Noah", "Alice", "Gabriel", "Mia",
                   "Félix", "Zoé", "Olivier", "Charlotte", "Antoine", "Rosalie"]
FR_LAST_NAMES = ["Tremblay", "Gagnon", "Roy", "Côté", "Bouchard", "Gauthier",
                  "Morin", "Lavoie", "Fortin", "Gagné", "Ouellet", "Pelletier"]

GAME_TYPES = ["Slots", "Video Poker", "Table Games", "Keno", "Bingo"]
GAME_TYPE_P = [0.68, 0.12, 0.10, 0.05, 0.05]
THEMES = ["Space", "Wildlife", "Mythology", "Pirates", "Egyptian", "Fantasy"]
MECHANICS = ["Hold & Spin", "Cascading Reels", "Free Spins", "Multiplier Wheel"]

# ─────────────────────────────────────────────────────────────────────────
# Geography -- Le Grand Palais is a Canadian chain: one property per city,
# spread across every province (Québec still the flagship market, Ontario
# second), plus a small Las Vegas cluster as its one US outpost. `intensity`
# is a per-city revenue multiplier tuned so the region hierarchy (QC > ON >
# rest of Canada > Vegas) actually shows up in generated bet volume, not just
# in city labels. `region` doubles as CrmLocationView.StateProv -- the
# Canadian province codes and "NV" all match the dashboard's existing map
# centroids, so the map "just works" against this data with no dashboard
# code changes.
# ─────────────────────────────────────────────────────────────────────────
CITIES = [
    # Québec -- flagship market (one property per city)
    dict(region="QC", country="CA", city="Montréal",         n=1, intensity=4.50, lat=45.5019, lon=-73.5674),
    dict(region="QC", country="CA", city="Québec City",      n=1, intensity=4.00, lat=46.8139, lon=-71.2080),
    dict(region="QC", country="CA", city="Percé",            n=1, intensity=2.50, lat=48.5238, lon=-64.2166),
    dict(region="QC", country="CA", city="Laval",             n=1, intensity=1.60, lat=45.5697, lon=-73.6923),
    dict(region="QC", country="CA", city="Gatineau",         n=1, intensity=1.30, lat=45.4765, lon=-75.7013),
    dict(region="QC", country="CA", city="Longueuil",        n=1, intensity=1.20, lat=45.5312, lon=-73.5183),
    dict(region="QC", country="CA", city="Sherbrooke",       n=1, intensity=1.20, lat=45.4042, lon=-71.8929),
    dict(region="QC", country="CA", city="Trois-Rivières",   n=1, intensity=1.20, lat=46.3432, lon=-72.5432),
    dict(region="QC", country="CA", city="Saguenay",         n=1, intensity=1.10, lat=48.4283, lon=-71.0686),
    dict(region="QC", country="CA", city="Lévis",            n=1, intensity=1.10, lat=46.8027, lon=-71.1778),
    dict(region="QC", country="CA", city="Drummondville",    n=1, intensity=1.00, lat=45.8835, lon=-72.4842),
    dict(region="QC", country="CA", city="Saint-Jérôme",     n=1, intensity=1.00, lat=45.7799, lon=-74.0028),
    dict(region="QC", country="CA", city="Granby",           n=1, intensity=0.90, lat=45.4009, lon=-72.7332),
    dict(region="QC", country="CA", city="Rimouski",         n=1, intensity=0.90, lat=48.4489, lon=-68.5230),
    dict(region="QC", country="CA", city="Val-d'Or",         n=1, intensity=0.80, lat=48.0977, lon=-77.7818),
    dict(region="QC", country="CA", city="Rouyn-Noranda",    n=1, intensity=0.80, lat=48.2359, lon=-79.0243),
    # Ontario -- strong secondary market
    dict(region="ON", country="CA", city="Ottawa",           n=1, intensity=3.50, lat=45.4215, lon=-75.6972),
    dict(region="ON", country="CA", city="Oakville",         n=1, intensity=3.30, lat=43.4675, lon=-79.6877),
    dict(region="ON", country="CA", city="Toronto",          n=1, intensity=1.60, lat=43.6532, lon=-79.3832),
    dict(region="ON", country="CA", city="Niagara Falls",    n=1, intensity=1.50, lat=43.0896, lon=-79.0849),
    dict(region="ON", country="CA", city="Windsor",          n=1, intensity=1.20, lat=42.3149, lon=-83.0364),
    dict(region="ON", country="CA", city="Hamilton",         n=1, intensity=1.20, lat=43.2557, lon=-79.8711),
    dict(region="ON", country="CA", city="Mississauga",      n=1, intensity=1.15, lat=43.5890, lon=-79.6441),
    dict(region="ON", country="CA", city="London",           n=1, intensity=1.10, lat=42.9849, lon=-81.2453),
    dict(region="ON", country="CA", city="Kitchener",        n=1, intensity=1.05, lat=43.4516, lon=-80.4925),
    dict(region="ON", country="CA", city="Kingston",         n=1, intensity=1.00, lat=44.2312, lon=-76.4860),
    dict(region="ON", country="CA", city="Sudbury",          n=1, intensity=1.00, lat=46.4917, lon=-80.9930),
    dict(region="ON", country="CA", city="Thunder Bay",      n=1, intensity=0.90, lat=48.3809, lon=-89.2477),
    dict(region="ON", country="CA", city="Barrie",           n=1, intensity=0.90, lat=44.3894, lon=-79.6903),
    dict(region="ON", country="CA", city="Guelph",           n=1, intensity=0.85, lat=43.5448, lon=-80.2482),
    dict(region="ON", country="CA", city="Oshawa",           n=1, intensity=0.85, lat=43.8971, lon=-78.8658),
    dict(region="ON", country="CA", city="St. Catharines",   n=1, intensity=0.85, lat=43.1594, lon=-79.2469),
    # British Columbia
    dict(region="BC", country="CA", city="Vancouver",        n=1, intensity=1.50, lat=49.2827, lon=-123.1207),
    dict(region="BC", country="CA", city="Victoria",         n=1, intensity=1.10, lat=48.4284, lon=-123.3656),
    dict(region="BC", country="CA", city="Surrey",           n=1, intensity=1.00, lat=49.1913, lon=-122.8490),
    dict(region="BC", country="CA", city="Kelowna",          n=1, intensity=0.90, lat=49.8880, lon=-119.4960),
    dict(region="BC", country="CA", city="Kamloops",         n=1, intensity=0.80, lat=50.6745, lon=-120.3273),
    dict(region="BC", country="CA", city="Nanaimo",          n=1, intensity=0.75, lat=49.1659, lon=-123.9401),
    # Alberta
    dict(region="AB", country="CA", city="Calgary",          n=1, intensity=1.50, lat=51.0447, lon=-114.0719),
    dict(region="AB", country="CA", city="Edmonton",         n=1, intensity=1.40, lat=53.5461, lon=-113.4938),
    dict(region="AB", country="CA", city="Red Deer",         n=1, intensity=0.85, lat=52.2681, lon=-113.8112),
    dict(region="AB", country="CA", city="Lethbridge",       n=1, intensity=0.80, lat=49.6956, lon=-112.8451),
    # Manitoba
    dict(region="MB", country="CA", city="Winnipeg",         n=1, intensity=1.20, lat=49.8951, lon=-97.1384),
    dict(region="MB", country="CA", city="Brandon",          n=1, intensity=0.75, lat=49.8483, lon=-99.9501),
    # Saskatchewan
    dict(region="SK", country="CA", city="Saskatoon",        n=1, intensity=1.00, lat=52.1332, lon=-106.6700),
    dict(region="SK", country="CA", city="Regina",           n=1, intensity=0.95, lat=50.4452, lon=-104.6189),
    # Nova Scotia
    dict(region="NS", country="CA", city="Halifax",          n=1, intensity=1.10, lat=44.6488, lon=-63.5752),
    dict(region="NS", country="CA", city="Sydney",           n=1, intensity=0.75, lat=46.1368, lon=-60.1942),
    # New Brunswick
    dict(region="NB", country="CA", city="Moncton",          n=1, intensity=0.90, lat=46.0878, lon=-64.7782),
    dict(region="NB", country="CA", city="Saint John",       n=1, intensity=0.85, lat=45.2733, lon=-66.0633),
    dict(region="NB", country="CA", city="Fredericton",      n=1, intensity=0.80, lat=45.9636, lon=-66.6431),
    # Prince Edward Island / Newfoundland and Labrador
    dict(region="PE", country="CA", city="Charlottetown",    n=1, intensity=0.70, lat=46.2382, lon=-63.1311),
    dict(region="NL", country="CA", city="St. John's",       n=1, intensity=0.85, lat=47.5615, lon=-52.7126),
    # United States -- the one non-Canadian outpost, a small flagship cluster in
    # Las Vegas only (not a broader US expansion)
    dict(region="NV", country="US", city="Las Vegas",        n=4, intensity=1.60, lat=36.1699, lon=-115.1398),
]

# Only Las Vegas has more than one property, so this is really just its 4 branch
# names -- real Vegas-area districts, for a touch of authenticity.
DISTRICTS = ["The Strip", "Downtown", "Summerlin", "Henderson"]

# ── Build LOCATIONS + per-location geography/intensity from CITIES ─────────
LOCATIONS: list[str] = []
LOC_META: dict[str, dict] = {}
_loc_i = 0
for c in CITIES:
    for k in range(c["n"]):
        loc_id = f"LOC{1000 + _loc_i}"
        jitter = float(rng.uniform(0.85, 1.15))
        name = f"Le Grand Palais – {c['city']}" if c["n"] == 1 else \
               f"Le Grand Palais – {c['city']} {DISTRICTS[k % len(DISTRICTS)]}"
        # Small jitter so multi-branch cities (only Las Vegas today) don't stack their
        # dots exactly on top of each other on the map.
        lat_j = c["lat"] + (rng.uniform(-0.06, 0.06) if c["n"] > 1 else 0.0)
        lon_j = c["lon"] + (rng.uniform(-0.06, 0.06) if c["n"] > 1 else 0.0)
        LOC_META[loc_id] = dict(
            region=c["region"], country=c["country"], city=c["city"],
            intensity=c["intensity"] * jitter, business_name=name,
            lat=lat_j, lon=lon_j,
        )
        LOCATIONS.append(loc_id)
        _loc_i += 1

N_LOCATIONS = len(LOCATIONS)
LOCATION_INTENSITY = {loc: LOC_META[loc]["intensity"] for loc in LOCATIONS}
_LOC_P = np.array([LOCATION_INTENSITY[loc] for loc in LOCATIONS])
_LOC_P = _LOC_P / _LOC_P.sum()

PFH_ACCOUNTS = [f"LGP-K{100000 + i}" for i in range(3500)]   # kiosk network loyalty accounts
EDGE_ACCOUNTS = [f"LGP-M{200000 + i}" for i in range(3500)]  # private members'-club accounts
LAND_ACCOUNTS = [f"LGP-{300000 + i}" for i in range(6000)]   # gaming-floor loyalty accounts

# Each property's pool of "regulars" -- players who show up in that property's
# floor data. Pulled from the shared LAND_ACCOUNTS pool so a player can be a
# regular at more than one property, same as a real multi-property loyalty program.
HOME_ACCOUNTS = {loc: rng.choice(LAND_ACCOUNTS, size=30, replace=False) for loc in LOCATIONS}

EDGE_CASINOS = sorted({f"{rng.choice(CASINO_ADJ)} {rng.choice(CASINO_NOUN)}" for _ in range(60)})[:28]


def _unique_names(n, rng_local):
    combos = [(a, b) for a in ADJ for b in NOUN]
    rng_local.shuffle(combos)
    out = []
    i = 0
    while len(out) < n:
        a, b = combos[i % len(combos)]
        suf = SUFFIX[(i // len(combos)) % len(SUFFIX)] if i >= len(combos) else rng_local.choice(SUFFIX)
        name = f"{a} {b}{suf}"
        if name not in out:
            out.append(name)
        i += 1
    return out


# ─────────────────────────────────────────────────────────────────────────
# 1. GameCatalogView1
# ─────────────────────────────────────────────────────────────────────────
def build_game_catalog():
    rows = []
    all_names = _unique_names(400, np.random.default_rng(SEED + 1))
    name_iter = iter(all_names)

    def gen_group(platform, id_start, n_base, n_hr, product_choices, hr_id_start):
        base_ids = list(range(id_start, id_start + n_base))
        recs = []
        for gid in base_ids:
            name = next(name_iter)
            recs.append(dict(
                Id=gid, Name=name,
                Type=rng.choice(GAME_TYPES, p=GAME_TYPE_P),
                Platform=platform,
                Product=rng.choice(product_choices["vals"], p=product_choices["p"]),
                Codebase=rng.choice(["gen0", "gen1", "gen2"], p=[0.2, 0.35, 0.45]),
                Status=rng.choice(["Live", "Live", "Live", "Live", "Discontinued", "Beta"]),
                MinBet=float(rng.choice([0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 5.0])),
                ScreenOrientation=rng.choice(["Horizontal", "Vertical", "Responsive"], p=[0.45, 0.45, 0.10]),
                JackpotStatus=rng.choice(["None", "Standalone", "Linked"], p=[0.6, 0.25, 0.15]),
                Vip=int(rng.random() < 0.08),
                Seasonal=int(rng.random() < 0.06),
                Mechanics=(rng.choice(MECHANICS) if rng.random() < 0.12 else None),
                Theme=(rng.choice(THEMES) if rng.random() < 0.12 else None),
                Branded=int(rng.random() < 0.05),
                SkinOf=None,
                ModifiedAt=(TODAY - dt.timedelta(days=int(rng.integers(0, 700)))),
                _launch_offset_days=int(rng.integers(7, 730)),
            ))
        # a few skin families among the base games (same platform)
        n_families = max(1, n_base // 12)
        for _ in range(n_families):
            parent, child = rng.choice(base_ids, size=2, replace=False)
            recs[base_ids.index(child)]["SkinOf"] = parent

        # HR ("high-roller" / test-rig) variants -- share a Name with a random base
        # game, mirroring the real system's confirmed "same name, 2 GameIds" pattern
        # (see reference_gameid_display memory) -- IDs land >= 95000 by design so the
        # dashboard's is_hr() >= 95000 filter has something real to exclude.
        hr_ids = list(range(hr_id_start, hr_id_start + n_hr))
        for i, gid in enumerate(hr_ids):
            parent = recs[i % len(recs)]
            recs.append(dict(
                Id=gid, Name=parent["Name"],
                Type=parent["Type"], Platform=platform, Product=parent["Product"],
                Codebase=parent["Codebase"], Status="Live",
                MinBet=parent["MinBet"], ScreenOrientation=parent["ScreenOrientation"],
                JackpotStatus="None", Vip=0, Seasonal=0, Mechanics=None, Theme=None,
                Branded=0, SkinOf=parent["Id"],
                ModifiedAt=(TODAY - dt.timedelta(days=int(rng.integers(0, 400)))),
                _launch_offset_days=int(rng.integers(7, 280)),
            ))
        return recs

    v1_products = {"vals": ["p2p", "sweeps", "pulltabs", "class2", "hhr", "gotskill"],
                   "p": [0.30, 0.30, 0.15, 0.10, 0.10, 0.05]}
    v2_products = {"vals": ["p2p", "sweeps", "pulltabs", "class2", "hhr"],
                   "p": [0.25, 0.30, 0.20, 0.15, 0.10]}
    ig_products = {"vals": ["pfh-edgelabs"], "p": [1.0]}

    rows += gen_group("v1", 1001, 55, 6, v1_products, 95001)
    rows += gen_group("v2", 2001, 55, 6, v2_products, 96001)
    rows += gen_group("igaming", 5001, 45, 0, ig_products, 97001)

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────
# 2. AnalyticsGameTerminalsGames  (+ TaskHandlerBetSpinSummary derived from it)
# ─────────────────────────────────────────────────────────────────────────
def _shape_curve(n_weeks, trend, rng_local):
    t = np.arange(n_weeks)
    ramp_w = min(6, n_weeks)
    # Week 0 starts at a real fraction of eventual scale (a newly installed game gets
    # real play from day one, not near-zero) and ramps the rest of the way to full scale
    # by ramp_w weeks. A near-0 week-0 floor here made "this week ÷ week 0" ratios (the
    # Bet Decay % KPI) explode to absurd four-figure percentages for almost every game,
    # since week 0 was always pinned to the same artificial floor regardless of trend --
    # this keeps the same ramp-up shape without that artifact.
    ramp = np.clip(0.45 + 0.55 * t / max(ramp_w, 1), 0.45, 1)
    if trend == "growing":
        base = ramp * np.clip(0.7 + 0.35 * t / max(n_weeks - 1, 1), 0, 1.3)
    elif trend == "declining":
        decline = np.clip(1 - 0.5 * (t - ramp_w) / max(n_weeks - ramp_w, 1), 0.25, 1)
        base = ramp * decline
    elif trend == "volatile":
        walk = np.cumsum(rng_local.normal(0, 0.06, n_weeks))
        base = ramp * np.clip(1 + walk - walk[0], 0.4, 1.5)
    else:  # stable
        base = ramp * np.clip(1 + rng_local.normal(0, 0.05, n_weeks), 0.7, 1.15)
    return np.clip(base, 0.2, None)


def build_terminal_games(catalog):
    landbased = catalog[catalog["Platform"].isin(["v1", "v2"])]
    all_rows = []
    for _, g in landbased.iterrows():
        gid = int(g["Id"])
        is_hr = gid >= 95000
        n_weeks = max(2, int(round(g["_launch_offset_days"] / 7)))
        launch_date = TODAY - dt.timedelta(days=int(n_weeks * 7))
        trend = rng.choice(["growing", "stable", "declining", "volatile"], p=[0.25, 0.4, 0.2, 0.15])
        curve = _shape_curve(n_weeks, trend, rng)

        if is_hr:
            peak_stores = int(rng.integers(3, 25))
            tier_bet = rng.uniform(60, 250)
        else:
            # Fewer/less-anemic small-tier games than a real fleet would have -- this is a
            # demo dataset meant to look good in every chart, not a realistic long tail.
            tier = rng.choice(["small", "medium", "hit"], p=[0.30, 0.45, 0.25])
            peak_stores = int({"small": rng.integers(12, 35), "medium": rng.integers(35, 130),
                                "hit": rng.integers(130, 400)}[tier])
            tier_bet = {"small": rng.uniform(180, 450), "medium": rng.uniform(400, 1000),
                        "hit": rng.uniform(1000, 2900)}[tier]
        hold_pct = rng.uniform(0.04, 0.12)
        avg_wager = rng.uniform(0.25, 2.0)
        pool_size = max(5, int(peak_stores * 1.3))
        loc_pool = rng.choice(LOCATIONS, size=min(pool_size, N_LOCATIONS), replace=False, p=_LOC_P)

        counts = np.maximum(1, np.round(peak_stores * curve * (1 + rng.normal(0, 0.07, n_weeks)))).astype(int)
        counts = np.minimum(counts, len(loc_pool))

        for w in range(n_weeks):
            c = int(counts[w])
            if c <= 0:
                continue
            week_start = launch_date + dt.timedelta(days=int(w * 7))
            if week_start > TODAY:
                break
            store_ids = rng.choice(loc_pool, size=c, replace=False)
            intens = np.array([LOCATION_INTENSITY[s] for s in store_ids])
            n_days = min(7, (TODAY - week_start).days + 1)
            for d in range(n_days):
                day = week_start + dt.timedelta(days=d)
                bet_dollars = tier_bet * (0.7 + 0.6 * curve[w]) * intens * (1 + rng.normal(0, 0.15, c))
                bet_dollars = np.clip(bet_dollars, 5, None)
                total_play_cents = np.round(bet_dollars * 100).astype(np.int64)
                total_win_cents = np.round(total_play_cents * (1 - hold_pct) *
                                            (1 + rng.normal(0, 0.05, c))).astype(np.int64)
                total_win_cents = np.clip(total_win_cents, 0, (total_play_cents * 1.3).astype(np.int64))
                play_count = np.maximum(1, np.round(bet_dollars / avg_wager)).astype(np.int64)
                player_accounts = [rng.choice(HOME_ACCOUNTS[s]) for s in store_ids]
                all_rows.append(pd.DataFrame({
                    "Id": gid, "SummaryDate": day, "TotalPlay": total_play_cents,
                    "TotalWin": total_win_cents, "PlayCount": play_count,
                    "SummaryLocationId": store_ids,
                    "PlayerAccountNumber": player_accounts,
                }))
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["Id", "SummaryDate", "TotalPlay", "TotalWin", "PlayCount", "SummaryLocationId",
                 "PlayerAccountNumber"])
    df["SummaryDate"] = pd.to_datetime(df["SummaryDate"])
    return df


def build_task_handler(terminal_games, catalog):
    """Trailing-30-day activity feed for the V1/V2 'active locations' + What's New
    Products lookup. Derived from the tail of AnalyticsGameTerminalsGames so it's
    always internally consistent with it."""
    cutoff = pd.Timestamp(TODAY - dt.timedelta(days=35))
    tail = terminal_games[terminal_games["SummaryDate"] >= cutoff].copy()
    if tail.empty:
        return pd.DataFrame(columns=["GameId", "StoreNumber", "SummaryDate", "Spins", "CasinoName", "ProductName"])
    prod_map = dict(zip(catalog["Id"], catalog["Product"]))
    tail["ProductName"] = tail["Id"].map(prod_map)
    out = pd.DataFrame({
        "GameId": tail["Id"],
        "StoreNumber": tail["SummaryLocationId"],
        "SummaryDate": tail["SummaryDate"],
        "Spins": tail["PlayCount"],
        "CasinoName": rng.choice(["vendor1", "vendor2"], size=len(tail)),
        "ProductName": tail["ProductName"],
    })
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# 3. BetSpinSummaryCashView3  (igaming: PFH kiosk network + EdgeLabs members' club)
# ─────────────────────────────────────────────────────────────────────────
def build_bet_spin_summary(catalog):
    igaming = catalog[catalog["Platform"] == "igaming"]
    home_store = {a: rng.choice(LOCATIONS, p=_LOC_P) for a in PFH_ACCOUNTS}
    home_casino = {a: rng.choice(EDGE_CASINOS) for a in EDGE_ACCOUNTS}
    all_rows = []

    for _, g in igaming.iterrows():
        gid = str(int(g["Id"]))
        n_weeks = max(2, int(round(g["_launch_offset_days"] / 7)))
        launch_date = TODAY - dt.timedelta(days=int(n_weeks * 7))
        trend = rng.choice(["growing", "stable", "declining", "volatile"], p=[0.25, 0.4, 0.2, 0.15])
        curve = _shape_curve(n_weeks, trend, rng)
        hold_pct = rng.uniform(0.04, 0.12)
        avg_wager = rng.uniform(0.10, 1.5)
        tier = rng.choice(["small", "medium", "hit"], p=[0.30, 0.45, 0.25])
        peak_players = int({"small": rng.integers(15, 45), "medium": rng.integers(45, 160),
                             "hit": rng.integers(160, 500)}[tier])
        tier_bet = {"small": rng.uniform(60, 150), "medium": rng.uniform(150, 420),
                    "hit": rng.uniform(420, 1250)}[tier]
        counts = np.maximum(1, np.round(peak_players * curve * (1 + rng.normal(0, 0.08, n_weeks)))).astype(int)

        on_pfh = rng.random() < 0.85
        on_edge = rng.random() < 0.55

        def emit(scope, account_pool, casino_source):
            for w in range(n_weeks):
                c = min(int(counts[w]), len(account_pool))
                if c <= 0:
                    continue
                week_start = launch_date + dt.timedelta(days=int(w * 7))
                if week_start > TODAY:
                    break
                accts = rng.choice(account_pool, size=c, replace=False)
                day_offsets = rng.integers(0, 7, size=c)
                days = [min(week_start + dt.timedelta(days=int(o)), TODAY) for o in day_offsets]
                if scope == "pfh":
                    stores = [home_store[a] for a in accts]
                    intens = np.array([LOCATION_INTENSITY[s] for s in stores])
                else:
                    stores = rng.choice(LOCATIONS, size=c)
                    intens = np.ones(c)
                bet_dollars = tier_bet * (0.7 + 0.6 * curve[w]) * intens * (1 + rng.normal(0, 0.4, c))
                bet_dollars = np.clip(bet_dollars, 2, None)
                total_bet_cents = np.round(bet_dollars * 100).astype(np.int64)
                total_win_cents = np.round(total_bet_cents * (1 - hold_pct) *
                                            (1 + rng.normal(0, 0.06, c))).astype(np.int64)
                total_win_cents = np.clip(total_win_cents, 0, (total_bet_cents * 1.3).astype(np.int64))
                spins = np.maximum(1, np.round(bet_dollars / avg_wager)).astype(np.int64)
                free_camp = np.where(rng.random(c) < 0.04, "CAMP" + rng.integers(1, 50, c).astype(str), None)
                if scope == "pfh":
                    casinos = ["PFH"] * c
                    platform_name = "Pong"
                else:
                    casinos = [home_casino[a] for a in accts]
                    platform_name = "EdgeLabs"
                all_rows.append(pd.DataFrame({
                    "PlatformName": platform_name, "CasinoName": casinos, "Date": days,
                    "StoreNumber": stores, "AccountNumber": accts, "GameId": gid,
                    "TotalBet": total_bet_cents, "TotalWin": total_win_cents, "Spins": spins,
                    "CurrencyName": rng.choice(["USD", "SC", "GC"], size=c,
                                                p=[0.7, 0.2, 0.1] if scope == "pfh" else [0.15, 0.55, 0.30]),
                    "AggregatorName": rng.choice(["AggOne", "AggTwo", "AggThree", None], size=c,
                                                  p=[0.35, 0.30, 0.20, 0.15]),
                    "FreeGameCampaignId": free_camp,
                    "PlayType": rng.choice(["Real", "Free"], size=c, p=[0.94, 0.06]),
                    "LifetimeSpinCount": rng.integers(1, 20000, c),
                    "ExtraPlay": rng.integers(0, 2, c),
                    "TotalReceivedWin": total_win_cents,
                    "TotalInstantWin": np.where(rng.random(c) < 0.02,
                                                 np.round(bet_dollars * 5 * 100).astype(np.int64), 0),
                }))

        if on_pfh:
            emit("pfh", PFH_ACCOUNTS, "PFH")
        if on_edge:
            emit("edge", EDGE_ACCOUNTS, None)
        if not on_pfh and not on_edge:
            emit("pfh", PFH_ACCOUNTS, "PFH")  # every igaming game lives somewhere

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ─────────────────────────────────────────────────────────────────────────
# 4. CrmLocationView
# ─────────────────────────────────────────────────────────────────────────
def build_crm_locations():
    rows = []
    platform_choices = ["PFH", "V2", "V1", "UNKNOWN"]
    product_by_platform = {
        "PFH": ["PFH + Sweeps", "Kiosk Only", "PFH Only"],
        "V2": ["P2P", "PullTabs", "Class 2", "HHR", "Sweeps"],
        "V1": ["Sweeps", "PFH Only", "P2P", "Got Skill"],
        "UNKNOWN": ["Sweeps"],
    }
    for i, loc_id in enumerate(LOCATIONS):
        meta = LOC_META[loc_id]
        cfg_platform = rng.choice(platform_choices, p=[0.35, 0.30, 0.25, 0.10])
        cfg_product = rng.choice(product_by_platform[cfg_platform])
        kiosk = int(cfg_platform == "PFH" and cfg_product in ("PFH + Sweeps", "Kiosk Only") and rng.random() < 0.3)
        if meta["region"] == "QC":
            mgr = f"{rng.choice(FR_FIRST_NAMES)} {rng.choice(FR_LAST_NAMES)}"
        else:
            mgr = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        rows.append(dict(
            LocationId=loc_id,
            StateProv=meta["region"],
            Country=meta["country"],
            City=meta["city"],
            Latitude=meta["lat"],
            Longitude=meta["lon"],
            BusinessName=meta["business_name"],
            Distributor=rng.choice(DISTRIBUTORS),
            Operator=rng.choice(OPERATORS),
            PFHEnabled=int(cfg_platform == "PFH"),
            ConfigProduct=cfg_product,
            ConfigStudio=rng.choice(STUDIOS),
            ConfigPlatform=cfg_platform,
            Kiosk=kiosk,
            AccountManager=mgr,
            **{"H-Wooden": int(rng.integers(0, 15)), "H-Metal": int(rng.integers(0, 10)),
               "BarTop": int(rng.integers(0, 8)), "V-Wooden": int(rng.integers(0, 15)),
               "V-Metal": int(rng.integers(0, 10)), "DualScreen": int(rng.integers(0, 6))},
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 5. CrmUpdateLogView  (+ LocationAnalyticsSummary for the Math Impact tab)
# ─────────────────────────────────────────────────────────────────────────
def build_crm_update_log(catalog, locations_df):
    rows = []
    landbased = catalog[catalog["Platform"].isin(["v1", "v2"])]
    igaming_pfh = catalog[catalog["Platform"] == "igaming"]

    # "Game" / "Enable" rows -- one simple (no '|' multi-game notes -- see the
    # load_game_releases() simplification note in launch_dashboard_v2.py) row group
    # per rollout location, spread over the weeks following each game's launch.
    for games_df, has_platform in ((landbased, True), (igaming_pfh, True)):
        for _, g in games_df.iterrows():
            n_weeks = max(1, int(round(g["_launch_offset_days"] / 7)))
            launch_date = TODAY - dt.timedelta(days=int(n_weeks * 7))
            if launch_date > TODAY:
                continue
            n_locs = int(rng.integers(3, 40))
            chosen_locs = rng.choice(LOCATIONS, size=min(n_locs, N_LOCATIONS), replace=False, p=_LOC_P)
            crm_platform = {"v1": rng.choice(["V1 Sweeps", "V1 Got Skill", "V1 Pay to Play"]),
                             "v2": rng.choice(["V2 Pay to Play", "V2 Pull-Tabs", "V2 Class 2", "V2 Sweeps"]),
                             "igaming": "PFH Sweeps"}[g["Platform"]]
            for loc in chosen_locs:
                enable_date = launch_date + dt.timedelta(days=int(rng.integers(0, 21)))
                rows.append(dict(LocationId=loc, Date=min(enable_date, TODAY), Category="Game",
                                  Action="Enable", Note=g["Name"], Platform=crm_platform))

    # A handful of Payout/Pool "math" update rows, for the Math Impact tab.
    # Lower-cased "math" substring deliberately -- load_math_update_groups()
    # filters with `Note LIKE '%math%'`, and unlike SQL Server's default
    # case-insensitive collation, DuckDB's LIKE is case-sensitive.
    math_notes = [f"math table update - {n}" for n in
                  rng.choice(catalog["Name"].unique(), size=min(20, catalog["Name"].nunique()), replace=False)]
    for note in math_notes:
        n_locs = int(rng.integers(5, 60))
        chosen_locs = rng.choice(LOCATIONS, size=min(n_locs, N_LOCATIONS), replace=False, p=_LOC_P)
        ref_date = TODAY - dt.timedelta(days=int(rng.integers(30, 500)))
        for loc in chosen_locs:
            d = ref_date + dt.timedelta(days=int(rng.integers(-3, 3)))
            rows.append(dict(LocationId=loc, Date=min(max(d, TODAY - dt.timedelta(days=729)), TODAY),
                              Category=rng.choice(["Payout", "Pool"]), Action="Update",
                              Note=note, Platform="PFH Sweeps"))

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_location_analytics_summary():
    rows = []
    n_weeks = 104
    week_starts = [TODAY - dt.timedelta(days=7 * w) for w in range(n_weeks)]
    week_starts = [w - dt.timedelta(days=w.weekday()) for w in week_starts]  # Monday
    for loc in LOCATIONS:
        base_play = rng.uniform(2000, 40000) * LOCATION_INTENSITY[loc]
        hold = rng.uniform(0.04, 0.12)
        for wk in week_starts:
            for d in range(7):
                day = wk + dt.timedelta(days=d)
                if day > TODAY:
                    continue
                play = max(0.0, base_play * (1 + rng.normal(0, 0.25)))
                win = play * (1 - hold) * (1 + rng.normal(0, 0.05))
                players = max(1, int(rng.poisson(play / 500 + 1)))
                rows.append((loc, day, wk, round(play, 2), round(win, 2), players))
    df = pd.DataFrame(rows, columns=["LocationId", "Date", "WeekStartingMonday",
                                      "TotalPlay", "TotalWin", "PlayerCount"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["WeekStartingMonday"] = pd.to_datetime(df["WeekStartingMonday"])
    return df


def _duckdb_safe(df: pd.DataFrame) -> pd.DataFrame:
    """pandas 3.x infers a native 'str' dtype for text columns that duckdb 1.1.3
    (predates pandas 3.0) can't register directly ('Data type str not recognized').
    Cast those columns back to classic numpy object dtype before handing to duckdb."""
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype) in ("str", "string"):
            df[col] = df[col].astype(object)
    return df


# ─────────────────────────────────────────────────────────────────────────
def main():
    print(f"Geography: {N_LOCATIONS} Le Grand Palais properties "
          f"({sum(1 for m in LOC_META.values() if m['region']=='QC')} QC, "
          f"{sum(1 for m in LOC_META.values() if m['region']=='ON')} ON, "
          f"{sum(1 for m in LOC_META.values() if m['country']=='US')} US)")

    print("Building GameCatalogView1 ...")
    catalog = build_game_catalog()
    print(f"  {len(catalog)} games")

    print("Building AnalyticsGameTerminalsGames ...")
    terminal_games = build_terminal_games(catalog)
    print(f"  {len(terminal_games):,} rows")

    print("Building TaskHandlerBetSpinSummary ...")
    task_handler = build_task_handler(terminal_games, catalog)
    print(f"  {len(task_handler):,} rows")

    print("Building BetSpinSummaryCashView3 (igaming) ...")
    bet_spin = build_bet_spin_summary(catalog)
    print(f"  {len(bet_spin):,} rows")

    print("Building CrmLocationView ...")
    locations = build_crm_locations()
    print(f"  {len(locations)} rows")

    print("Building CrmUpdateLogView ...")
    update_log = build_crm_update_log(catalog, locations)
    print(f"  {len(update_log):,} rows")

    print("Building LocationAnalyticsSummary ...")
    loc_summary = build_location_analytics_summary()
    print(f"  {len(loc_summary):,} rows")

    catalog_out = _duckdb_safe(catalog.drop(columns=["_launch_offset_days"]))
    terminal_games = _duckdb_safe(terminal_games)
    task_handler = _duckdb_safe(task_handler)
    bet_spin = _duckdb_safe(bet_spin)
    locations = _duckdb_safe(locations)
    update_log = _duckdb_safe(update_log)
    loc_summary = _duckdb_safe(loc_summary)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = duckdb.connect(DB_PATH)

    con.register("catalog_out", catalog_out)
    con.execute("CREATE TABLE GameCatalogView1 AS SELECT * FROM catalog_out")

    con.register("terminal_games", terminal_games)
    con.execute("CREATE TABLE AnalyticsGameTerminalsGames AS SELECT * FROM terminal_games")

    con.register("task_handler", task_handler)
    con.execute("CREATE TABLE TaskHandlerBetSpinSummary AS SELECT * FROM task_handler")

    con.register("bet_spin", bet_spin)
    con.execute("CREATE TABLE BetSpinSummaryCashView3 AS SELECT * FROM bet_spin")

    con.execute("""CREATE VIEW BetSpinSummaryCashView3Pong AS
                   SELECT * FROM BetSpinSummaryCashView3
                   WHERE PlatformName='Pong' AND CasinoName='PFH'""")
    con.execute("""CREATE VIEW BetSpinSummaryCashView3EdgeLabs AS
                   SELECT * FROM BetSpinSummaryCashView3
                   WHERE PlatformName='EdgeLabs'""")
    con.execute("""CREATE VIEW BetSpinSummarySocialView2 AS
                   SELECT * FROM BetSpinSummaryCashView3
                   WHERE PlatformName='EdgeLabs'""")

    con.register("locations", locations)
    con.execute("CREATE TABLE CrmLocationView AS SELECT * FROM locations")

    con.register("update_log", update_log)
    con.execute("CREATE TABLE CrmUpdateLogView AS SELECT * FROM update_log")

    con.register("loc_summary", loc_summary)
    con.execute("CREATE TABLE LocationAnalyticsSummary AS SELECT * FROM loc_summary")

    con.close()
    print(f"\nWrote {DB_PATH}")


if __name__ == "__main__":
    main()
