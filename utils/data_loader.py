"""Data loading utilities for market prioritisation system."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

DATA_DIR = Path(__file__).parent.parent / "data"


def load_markets() -> List[Dict[str, Any]]:
    """Load the APAC market universe."""
    with open(DATA_DIR / "markets.json") as f:
        return json.load(f)


def load_demand_data() -> pd.DataFrame:
    """Load demand signal data keyed by market_code."""
    df = pd.read_csv(DATA_DIR / "demand_data.csv")
    df.set_index("market_code", inplace=True)
    return df


def load_supply_data() -> pd.DataFrame:
    """Load competitive supply data keyed by market_code."""
    df = pd.read_csv(DATA_DIR / "supply_data.csv")
    df.set_index("market_code", inplace=True)
    return df


def load_macro_data() -> pd.DataFrame:
    """Load macro & regulatory data keyed by market_code."""
    df = pd.read_csv(DATA_DIR / "macro_data.csv")
    df.set_index("market_code", inplace=True)
    return df


def get_market_package(market_code: str) -> Dict[str, Any]:
    """Return all data for a single market as a consolidated dict."""
    demand_df = load_demand_data()
    supply_df = load_supply_data()
    macro_df = load_macro_data()
    markets = {m["code"]: m for m in load_markets()}

    market_info = markets.get(market_code, {})

    demand = (
        demand_df.loc[market_code].to_dict()
        if market_code in demand_df.index
        else {}
    )
    supply = (
        supply_df.loc[market_code].to_dict()
        if market_code in supply_df.index
        else {}
    )
    macro = (
        macro_df.loc[market_code].to_dict()
        if market_code in macro_df.index
        else {}
    )

    return {
        "market_code": market_code,
        "market_info": market_info,
        "demand": demand,
        "supply": supply,
        "macro": macro,
    }


def get_all_market_packages() -> List[Dict[str, Any]]:
    """Return full data packages for all markets."""
    markets = load_markets()
    return [get_market_package(m["code"]) for m in markets]
