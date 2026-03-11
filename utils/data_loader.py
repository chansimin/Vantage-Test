"""Data loading utilities — supports both city-level and sub-market level data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# City-level helpers (legacy + macro)
# ---------------------------------------------------------------------------

def load_markets() -> List[Dict[str, Any]]:
    with open(DATA_DIR / "markets.json") as f:
        return json.load(f)


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def get_market_package(market_code: str) -> Dict[str, Any]:
    markets   = {m["code"]: m for m in load_markets()}
    demand_df = _csv("demand_data.csv").set_index("market_code")
    supply_df = _csv("supply_data.csv").set_index("market_code")
    macro_df  = _csv("macro_data.csv").set_index("market_code")
    return {
        "market_code": market_code,
        "market_info": markets.get(market_code, {}),
        "demand": demand_df.loc[market_code].to_dict() if market_code in demand_df.index else {},
        "supply": supply_df.loc[market_code].to_dict() if market_code in supply_df.index else {},
        "macro":  macro_df.loc[market_code].to_dict()  if market_code in macro_df.index  else {},
    }


def get_all_market_packages() -> List[Dict[str, Any]]:
    return [get_market_package(m["code"]) for m in load_markets()]


# ---------------------------------------------------------------------------
# Sub-market helpers
# ---------------------------------------------------------------------------

def load_submarkets() -> List[Dict[str, Any]]:
    with open(DATA_DIR / "submarkets.json") as f:
        return json.load(f)


def get_all_submarket_packages() -> List[Dict[str, Any]]:
    submarkets = {sm["code"]: sm for sm in load_submarkets()}
    cloud_df   = _csv("cloud_data.csv").set_index("submarket_code")
    ai_df      = _csv("ai_data.csv").set_index("submarket_code")
    macro_df   = _csv("macro_data.csv").set_index("market_code")

    packages = []
    for code, info in submarkets.items():
        city_code = info.get("city_code", "")
        packages.append({
            "submarket_code": code,
            "submarket_info": info,
            "cloud": cloud_df.loc[code].to_dict()     if code      in cloud_df.index else {},
            "ai":    ai_df.loc[code].to_dict()         if code      in ai_df.index    else {},
            "macro": macro_df.loc[city_code].to_dict() if city_code in macro_df.index else {},
        })
    return packages


def get_submarket_package(submarket_code: str) -> Dict[str, Any]:
    submarkets = {sm["code"]: sm for sm in load_submarkets()}
    cloud_df   = _csv("cloud_data.csv").set_index("submarket_code")
    ai_df      = _csv("ai_data.csv").set_index("submarket_code")
    macro_df   = _csv("macro_data.csv").set_index("market_code")
    info       = submarkets.get(submarket_code, {})
    city_code  = info.get("city_code", "")
    return {
        "submarket_code": submarket_code,
        "submarket_info": info,
        "cloud": cloud_df.loc[submarket_code].to_dict() if submarket_code in cloud_df.index else {},
        "ai":    ai_df.loc[submarket_code].to_dict()    if submarket_code in ai_df.index    else {},
        "macro": macro_df.loc[city_code].to_dict()      if city_code in macro_df.index      else {},
    }
