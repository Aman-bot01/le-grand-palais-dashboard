# Le Grand Palais — Casino Intelligence

An analytics dashboard for **Le Grand Palais**, a fictional physical casino
chain: one property per city, spread across every Canadian province (Québec
the flagship market, Ontario second) plus a small Las Vegas outpost — 57
properties in total.

Every game, property, player account, and revenue figure in this app is
**synthetically generated** — there is no real company data anywhere in it.
Safe to run, share, screenshot, demo, or deploy publicly.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run launch_dashboard_v2.py --server.port 8504
```

That's it — `synthetic.duckdb` builds itself automatically on first run if it
isn't already present (takes under a minute). To force a fresh dataset, delete
`synthetic.duckdb` and start the app again; `synthetic_data.py` is seeded, so
a rebuild reproduces the same dataset every time.

## What's inside

| File | What it is |
|---|---|
| `launch_dashboard_v2.py` | The dashboard UI — 10 tabs |
| `launch.py` | Analytics engine — DTW peer matching, Quick Score, forecasting |
| `engine.py` | Connects to local DuckDB (auto-builds it on first run) and registers SQL-Server-compatibility macros (`GETDATE`, `ISNULL`, `DATEDIFF`, `DATEADD`) so the T-SQL-style query strings elsewhere run mostly unmodified |
| `synthetic_data.py` | Generates the entire dataset — games, properties, players, revenue |
| `synthetic.duckdb` | The generated database (not committed — built on first run; ~30 MB, ~2.5M rows) |

## The dataset

~180 games across four internal "systems" (routing detail only — the
sidebar just shows one combined game list, no platform picker) and 57
Le Grand Palais properties, one per city: 15 in Québec, 15 in Ontario, a
further 25 spread across every other Canadian province, and a 4-branch Las
Vegas cluster as the one US outpost. Revenue intensity is tuned per-city so
Québec genuinely out-earns Ontario, which out-earns the rest — not just in
labels, in the actual generated numbers. Player-level loyalty-account data
(tiered into New/Casual, Regular, Premium, VIP segments) exists for every
floor, including the physical gaming floors. Launch dates span about two
years with varied performance so the dashboard has something interesting to
show: strong performers, decliners, and brand-new launches.

## Tabs

All Games Health · Weekly Games · What's New · Location Overview ·
Is It On Track? · Similar Launches & What to Expect · Full Breakdown ·
Loyalty & Members · Compare · Game Clusters

## Verified working

All 10 tabs render with content and zero exceptions, in both "All Games" and
single-game mode, across every underlying game system. Peer matching, Quick
Score, forecasting, the property map, and `.docx` report generation all run
successfully end to end.
