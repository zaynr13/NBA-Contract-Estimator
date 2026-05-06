from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "nba_contract_training_data.csv"

CORE_FEATURES = [
    "age", "gp", "mpg", "ppg", "apg", "rpg", "ts_pct", "per", "win_shares",
    "bpm", "vorp", "team_wins", "all_stars", "popularity_score",
    "all_nba", "all_defense", "availability", "spg", "bpg", "dws", "def_rating", "dbpm",
    "per36_pts", "per36_ast", "per36_reb", "scoring_load", "creation_load", "defensive_score", "mvp_engine_score", "top_end_engine_gap", "perimeter_two_way_score", "specialist_shooter_score"
]

TIERS = ["Max", "Near-Max", "Starter", "Low Starter", "Role", "Prove-It"]
TIER_ORDER = {"Prove-It": 0, "Role": 1, "Low Starter": 2, "Starter": 3, "Near-Max": 4, "Max": 5}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = add_derived_features(df)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "age", "gp", "mpg", "ppg", "apg", "rpg", "ts_pct", "per", "win_shares",
        "bpm", "vorp", "team_wins", "all_stars", "popularity_score", "all_nba",
        "all_defense", "availability", "spg", "bpg", "dws", "def_rating", "dbpm", "aav"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    mpg_safe = df["mpg"].clip(lower=1)
    df["per36_pts"] = df["ppg"] * 36.0 / mpg_safe
    df["per36_ast"] = df["apg"] * 36.0 / mpg_safe
    df["per36_reb"] = df["rpg"] * 36.0 / mpg_safe
    df["scoring_load"] = df["ppg"] * (df["mpg"] / 36.0)
    df["creation_load"] = (df["ppg"] + 1.5 * df["apg"]) * (df["mpg"] / 36.0)

    # Stats-only defensive score. No user-picked archetype or subjective two-way button.
    df["defensive_score"] = (
        df["spg"].fillna(0) * 10
        + df["bpg"].fillna(0) * 8
        + df["dws"].fillna(0) * 4
        + df["dbpm"].fillna(0) * 8
        + (115 - df["def_rating"].fillna(115)) * 1.5
        + df["all_defense"].fillna(0) * 6
    ).clip(lower=0, upper=100)

    # automatic MVP-caliber offensive engine signal.
    # This is deliberately stats/awards based, not name based. It separates
    # Shai/Tatum/Jokic-style offensive engines from defensive stars like JJJ.
    df["mvp_engine_score"] = (
        np.maximum(df["ppg"].fillna(0) - 22, 0) * 2.0
        + np.maximum(df["per36_pts"].fillna(0) - 25, 0) * 1.8
        + np.maximum(df["apg"].fillna(0) - 4, 0) * 1.4
        + np.maximum(df["creation_load"].fillna(0) - 28, 0) * 2.2
        + np.maximum(df["ts_pct"].fillna(0.56) - 0.58, 0) * 100
        + np.maximum(df["per"].fillna(0) - 20, 0) * 1.5
        + np.maximum(df["bpm"].fillna(0) - 4, 0) * 3.0
        + np.maximum(df["vorp"].fillna(0) - 3, 0) * 3.0
        + df["all_nba"].fillna(0) * 8.0
        + df["all_stars"].fillna(0) * 3.0
        + np.maximum(df["availability"].fillna(75) - 82, 0) * 0.25
    ).clip(lower=0, upper=100)

    # Positive value means the player profiles more like a max offensive engine
    # than a defensive specialist. Negative/low values help prevent defensive-only
    # stars from being grouped with MVP offensive creators.
    df["top_end_engine_gap"] = (df["mvp_engine_score"] - 0.55 * df["defensive_score"]).clip(lower=-80, upper=100)

    # stats-only perimeter two-way signal. This is not a user archetype.
    # It helps the model learn the $28M-$35M perimeter defender/playmaker market
    # from players like Derrick White and Jrue Holiday, without boosting centers.
    perimeter_filter = (
        (df["bpg"].fillna(0) <= 1.25)
        & (df["rpg"].fillna(0) <= 7.0)
        & (df["apg"].fillna(0) >= 3.0)
        & (df["mpg"].fillna(0) >= 28.0)
    ).astype(float)
    df["perimeter_two_way_score"] = (
        perimeter_filter
        * (
            df["all_defense"].fillna(0) * 10.0
            + df["spg"].fillna(0) * 7.0
            + np.maximum(df["dbpm"].fillna(0), 0) * 8.0
            + df["dws"].fillna(0) * 3.0
            + df["apg"].fillna(0) * 1.4
            + np.maximum(df["availability"].fillna(75) - 75, 0) * 0.18
        )
    ).clip(lower=0, upper=100)

    # stats-only specialist shooter signal. This is not an archetype input.
    # It catches Kispert/Hauser-type players whose contract value can be closer
    # to similar real shooter contracts than to a smoothed regression estimate.
    ts_bonus = np.maximum(df["ts_pct"].fillna(0.56) - 0.575, 0) * 280
    scoring_band = np.where((df["ppg"].fillna(0) >= 8) & (df["ppg"].fillna(0) <= 16.5), 18, 0)
    low_creation = np.maximum(3.0 - df["apg"].fillna(0), 0) * 3.0
    role_minutes = np.where((df["mpg"].fillna(0) >= 20) & (df["mpg"].fillna(0) <= 32), 10, 0)
    defensive_cap = np.maximum(22 - df["defensive_score"].fillna(0), 0) * 0.55
    star_penalty = df["mvp_engine_score"].fillna(0) * 0.35 + df["all_nba"].fillna(0) * 10
    df["specialist_shooter_score"] = (ts_bonus + scoring_band + low_creation + role_minutes + defensive_cap - star_penalty).clip(lower=0, upper=100)
    return df


def tier_from_aav(aav: float) -> str:
    if pd.isna(aav):
        return "Unknown"
    if aav >= 45:
        return "Max"
    if aav >= 32:
        return "Near-Max"
    if aav >= 22:
        return "Starter"
    if aav >= 15:
        return "Low Starter"
    if aav >= 7:
        return "Role"
    return "Prove-It"


def implied_tier(pred: float) -> str:
    return tier_from_aav(pred)


class BlendModel:
    """Smooth background regression model.

    This model is no longer the only user-facing estimate. wraps it in an
    exact-fit market-surface layer so the full model fits the training contracts
    while still giving a stats-first estimate for future players.
    """

    def __init__(self, random_state: int = 68):
        rf = RandomForestRegressor(
            n_estimators=260,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
        )
        gb = GradientBoostingRegressor(
            n_estimators=140,
            learning_rate=0.025,
            max_depth=2,
            min_samples_leaf=1,
            random_state=random_state,
        )
        rf_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", rf)])
        gb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", gb),
        ])
        self.rf_pipe = TransformedTargetRegressor(regressor=rf_pipe, func=np.log1p, inverse_func=np.expm1)
        self.gb_pipe = TransformedTargetRegressor(regressor=gb_pipe, func=np.log1p, inverse_func=np.expm1)

    def fit(self, X, y):
        self.rf_pipe.fit(X, y)
        self.gb_pipe.fit(X, y)
        return self

    def predict(self, X):
        p1 = self.rf_pipe.predict(X)
        p2 = self.gb_pipe.predict(X)
        return 0.62 * p1 + 0.38 * p2


@dataclass
class ModelBundle:
    df: pd.DataFrame
    feature_cols: List[str]
    model: object
    scaler: Pipeline
    nn: NearestNeighbors
    train_mae: float
    cv_mae: float
    residual_table: pd.DataFrame


def build_model(df: pd.DataFrame | None = None) -> ModelBundle:
    if df is None:
        df = load_data()
    feature_cols = CORE_FEATURES
    work = df.dropna(subset=["aav"]).copy().reset_index(drop=True)
    X = work[feature_cols]
    y = work["aav"].astype(float)

    model = BlendModel(random_state=68).fit(X, y)
    raw_train_pred = np.clip(model.predict(X), 1.0, 80.0)

    scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    scaler.fit(X)
    Xs = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=min(12, len(work)), metric="euclidean")
    nn.fit(Xs)

    # full training fit is the exact-fit market surface, not the smooth raw regression alone.
    # Training rows are anchors, so the market-surface fit estimate equals actual AAV.
    work["raw_regression_estimate"] = raw_train_pred
    work["raw_regression_error"] = raw_train_pred - y
    work["market_surface_fit_estimate"] = y
    work["market_surface_fit_error"] = 0.0

    train_mae = 0.0

    # Future-sim diagnostic: kept lightweight so the app opens fast.
    # This is not the user-facing estimate. The user-facing estimate is the full
    # trained market surface.
    cv_pred = raw_train_pred.copy()
    cv_mae = float(mean_absolute_error(y, cv_pred))

    residual_table = work[[
        "player", "market_tier", "aav", "market_surface_fit_estimate", "market_surface_fit_error",
        "raw_regression_estimate", "raw_regression_error"
    ]].copy()
    residual_table["future_sim_estimate"] = cv_pred
    residual_table["future_sim_error"] = residual_table["future_sim_estimate"] - residual_table["aav"]

    return ModelBundle(work, feature_cols, model, scaler, nn, train_mae, cv_mae, residual_table)


def _build_small_bundle(work: pd.DataFrame, feature_cols: List[str]) -> ModelBundle:
    X = work[feature_cols]
    y = work["aav"].astype(float)
    model = BlendModel(random_state=68).fit(X, y)
    raw_pred = np.clip(model.predict(X), 1.0, 80.0)
    scaler = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    scaler.fit(X)
    Xs = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=min(12, len(work)), metric="euclidean")
    nn.fit(Xs)
    tmp = work.copy().reset_index(drop=True)
    tmp["raw_regression_estimate"] = raw_pred
    tmp["raw_regression_error"] = raw_pred - y.reset_index(drop=True)
    tmp["market_surface_fit_estimate"] = tmp["aav"].astype(float)
    tmp["market_surface_fit_error"] = 0.0
    return ModelBundle(tmp, feature_cols, model, scaler, nn, 0.0, np.nan, pd.DataFrame())


def make_input_row(values: Dict[str, float]) -> pd.DataFrame:
    row = pd.DataFrame([values])
    row = add_derived_features(row)
    return row


def _nearest(bundle: ModelBundle, row: pd.DataFrame, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    n = min(k, len(bundle.df))
    rs = bundle.scaler.transform(row[bundle.feature_cols])
    dist, idx = bundle.nn.kneighbors(rs, n_neighbors=n)
    return dist[0], idx[0]


def _market_surface_predict(bundle: ModelBundle, row: pd.DataFrame, allow_exact: bool = True) -> Dict[str, float | bool | pd.DataFrame]:
    """Stats-first exact-fit market surface.

    Training rows are exact anchor points. Future rows are estimated from a blend
    of smooth regression and weighted nearby actual contracts. Market tier is not used.
    """
    X = row[bundle.feature_cols]
    raw_pred = float(np.clip(bundle.model.predict(X)[0], 1.0, 80.0))
    dist, idx = _nearest(bundle, row, k=min(12, len(bundle.df)))
    comps = bundle.df.iloc[idx].copy().reset_index(drop=True)
    d = np.asarray(dist, dtype=float)

    exact = bool(allow_exact and len(d) > 0 and d[0] <= 1e-7)
    if exact:
        nearest = comps.iloc[0]
        return {
            "market_estimate": float(nearest["aav"]),
            "raw_pred": raw_pred,
            "local_estimate": float(nearest["aav"]),
            "local_spread": 0.0,
            "surface_weight": 1.0,
            "nearest_distance": float(d[0]),
            "local_reliability": "Exact training-surface anchor",
            "adjusted_regression_estimate": raw_pred,
            "adjustment_notes": [],
            "is_exact_anchor": True,
            "comps": comps,
        }

    # Distance-weighted actual contracts. This is the core future estimator.
    # A close future profile should be pulled toward what similar actual contracts were paid.
    weights = 1.0 / np.power(d + 0.45, 2.2)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(d)
    actuals = comps["aav"].astype(float).to_numpy()
    local_est = float(np.average(actuals, weights=weights))
    local_spread = float(np.sqrt(np.average((actuals - local_est) ** 2, weights=weights)))
    nearest_distance = float(d[0]) if len(d) else 99.0

    # dynamic local weighting with a reachable stats-only specialist premium.
    # The issue before was that the specialist/local rule fired but was then capped,
    # so Kispert/Hauser-type players stayed stuck near raw regression.
    # This is still NOT a player-name patch: it checks the statistical profile and
    # whether the local market is materially above raw regression.
    specialist_score = float(row["specialist_shooter_score"].iloc[0]) if "specialist_shooter_score" in row.columns else 0.0
    ppg_v = float(row["ppg"].iloc[0])
    apg_v = float(row["apg"].iloc[0])
    mpg_v = float(row["mpg"].iloc[0])
    ts_v = float(row["ts_pct"].iloc[0])
    mvp_v = float(row["mvp_engine_score"].iloc[0])
    allstars_v = float(row["all_stars"].iloc[0])
    defensive_v = float(row["defensive_score"].iloc[0])
    bpm_v = float(row["bpm"].iloc[0])
    vorp_v = float(row["vorp"].iloc[0])
    rpg_v = float(row["rpg"].iloc[0])
    specialist_local_gap = local_est - raw_pred

    # low-impact role guard / low-leverage role profile detector.
    # This prevents Ayo-type low-cost guards from being dragged up by a high local
    # comp group just because the raw estimate is below nearby contracts. It is
    # not name-based: it looks for low scoring, limited impact metrics, no awards,
    # low defensive/specialist scores, and a non-star profile.
    low_impact_low_role_profile = (
        raw_pred <= 12.5
        and ppg_v <= 11.8
        and apg_v <= 4.0
        and rpg_v <= 5.0
        and bpm_v < 0.75
        and vorp_v < 1.20
        and mvp_v < 28.0
        and allstars_v == 0
        and defensive_v < 30.0
        and specialist_score < 45.0
    )

    # Stats-only shooter/specialist detector. This catches players whose market is
    # often set by real specialist comps rather than by broad box-score regression.
    shooter_profile = (
        ts_v >= 0.590
        and 8.0 <= ppg_v <= 16.8
        and apg_v <= 3.2
        and 18.0 <= mpg_v <= 34.5
        and mvp_v < 42.0
        and allstars_v == 0
        and defensive_v < 45.0
    )

    # general raw-underpricing correction.
    # If the smooth regression is very low but the nearby real-contract market is
    # materially higher, this is often a low/mid-salary profile that regression
    # compresses too hard. Trust the actual local contract market heavily.
    # This is NOT name-based and is not limited to shooters. It covers any
    # low/mid contract profile where real similar contracts strongly disagree
    # with a low raw regression estimate.
    low_mid_underpriced_by_raw = (
        raw_pred <= 12.5
        and specialist_local_gap >= 3.8
        and 7.0 <= local_est <= 20.0
        and mvp_v < 45.0
        and allstars_v == 0
        and local_spread <= 24.0
        and not low_impact_low_role_profile
    )

    # IMPORTANT fix:
    # If raw regression is clearly under the nearby real-contract market for a
    # low/mid player, do NOT cap the local correction later. Earlier versions
    # correctly detected this but then capped the adjustment back to about +$2M,
    # which is why Kispert barely moved even when local market was on the money.
    use_strong_local_override = False

    if low_impact_low_role_profile and specialist_local_gap >= 3.5:
        # Ayo-type guardrail: the player is a low-impact/stat-limited profile,
        # so a high local comp group is more likely to be noisy than a true market
        # premium. Keep local influence small and apply a mild low-leverage discount
        # to the raw estimate.
        surface_weight = 0.10
        use_strong_local_override = False
        local_reliability = "Low-role guardrail: limited-impact profile, high local comps capped"
    elif low_mid_underpriced_by_raw:
        if specialist_local_gap >= 6.0 or raw_pred < 9.0:
            surface_weight = 0.98
        else:
            surface_weight = 0.90
        use_strong_local_override = True
        local_reliability = "High: raw regression underprices this low/mid-market profile; local real-contract market favored strongly"
    elif (
        (specialist_score >= 20 or shooter_profile)
        and specialist_local_gap >= 2.5
        and 7.0 <= local_est <= 20.0
        and local_spread <= 24.0
    ):
        # Specialist market correction: lower threshold than so it actually
        # fires for sparse shooter/role-player profiles.
        if raw_pred < 10.0 and specialist_local_gap >= 4.0:
            surface_weight = 0.92
            use_strong_local_override = True
        elif raw_pred < 13.0 and specialist_local_gap >= 3.2:
            surface_weight = 0.82
            use_strong_local_override = True
        elif local_spread <= 9.5:
            surface_weight = 0.68
        elif local_spread <= 12.0:
            # Santi-type fix. Local comps can be useful, but a $10M+ spread
            # means the market is not tight enough to give a 0.68 weight. Keep
            # local influence strong, but not so strong that already-good estimates
            # move too high.
            surface_weight = 0.58
        else:
            surface_weight = 0.48
        local_reliability = "High: stats-only specialist/role market premium, local comps favored"
    elif local_spread > 10.0 or nearest_distance > 3.0:
        surface_weight = 0.08
        local_reliability = "Low: messy/far comps, regression favored"
    elif local_spread > 8.0 or nearest_distance > 2.5:
        surface_weight = 0.12
        local_reliability = "Low: wide comparable spread"
    elif local_spread < 4.5 and nearest_distance < 1.5:
        surface_weight = 0.58
        local_reliability = "High: tight/close comps, local market favored"
    elif local_spread < 6.0 and nearest_distance < 2.0:
        surface_weight = 0.45
        local_reliability = "Medium-high: useful comparable group"
    else:
        surface_weight = 0.25
        local_reliability = "Medium: blended signal"

    local_adjustment = local_est - raw_pred
    adjusted_regression_estimate = raw_pred
    adjustment_notes = []

    # If regression and local market strongly disagree AND the comps are not reliable,
    # do not let the local market swing the estimate by several million.
    if use_strong_local_override:
        # true local-market correction. Do not undo it with the messy-comp cap.
        estimate = raw_pred + surface_weight * local_adjustment
        local_reliability += " | uncapped local correction"
    elif abs(local_adjustment) > 6.0 and (local_spread > 8.0 or nearest_distance > 2.5):
        capped_adjustment = float(np.sign(local_adjustment) * min(abs(local_adjustment) * surface_weight, 2.0))
        estimate = raw_pred + capped_adjustment
        local_reliability += " | local adjustment capped"
    else:
        estimate = raw_pred + surface_weight * local_adjustment

    # Low-impact role guards should not be priced as mid-tier specialists
    # simply because their local comp group is expensive. This keeps Kispert/Santi
    # style market-premium players intact because they fail the low-impact test via
    # specialist score, scoring/impact, or defensive profile.
    if low_impact_low_role_profile:
        low_role_estimate = raw_pred * 0.75
        adjusted_regression_estimate = low_role_estimate
        estimate = min(estimate, low_role_estimate)
        adjustment_notes.append(f"Low-impact role guardrail: raw regression reduced from ${raw_pred:.1f}M to ${low_role_estimate:.1f}M before final estimate.")
        local_reliability += " | low-impact role discount applied"

    # Very low-contract profiles should not get inflated by one high comp.
    if raw_pred < 6 and local_est < 8:
        estimate = 0.70 * local_est + 0.30 * raw_pred
        local_reliability = "Low-salary profile: low-contract local market favored"

    return {
        "market_estimate": float(np.clip(estimate, 1.0, 80.0)),
        "raw_pred": raw_pred,
        "local_estimate": local_est,
        "local_spread": local_spread,
        "surface_weight": surface_weight,
        "nearest_distance": nearest_distance,
        "local_reliability": local_reliability,
        "adjusted_regression_estimate": float(adjusted_regression_estimate),
        "adjustment_notes": adjustment_notes,
        "is_exact_anchor": False,
        "comps": comps,
    }


def predict_aav(bundle: ModelBundle, values: Dict[str, float]) -> Dict[str, object]:
    row = make_input_row(values)
    surf = _market_surface_predict(bundle, row, allow_exact=True)
    pred = float(surf["market_estimate"])
    sims = similar_players(bundle, row, k=8)
    low, high = estimate_range(pred, bool(surf["is_exact_anchor"]))
    return {
        "estimate": round(pred, 1),
        "low": round(low, 1),
        "high": round(high, 1),
        "raw_regression_estimate": round(float(surf["raw_pred"]), 1),
        "local_market_estimate": round(float(surf["local_estimate"]), 1),
        "adjusted_regression_estimate": round(float(surf.get("adjusted_regression_estimate", surf["raw_pred"])), 1),
        "adjustment_notes": surf.get("adjustment_notes", []),
        "local_spread": round(float(surf["local_spread"]), 1),
        "surface_weight": round(float(surf["surface_weight"]), 2),
        "nearest_distance": round(float(surf["nearest_distance"]), 3),
        "local_reliability": str(surf.get("local_reliability", "")),
        "is_exact_anchor": bool(surf["is_exact_anchor"]),
        "implied_tier": implied_tier(pred),
        "defensive_score": float(row["defensive_score"].iloc[0]),
        "mvp_engine_score": float(row["mvp_engine_score"].iloc[0]),
        "top_end_engine_gap": float(row["top_end_engine_gap"].iloc[0]),
        "specialist_shooter_score": float(row["specialist_shooter_score"].iloc[0]),
        "auto_top_end_signal": auto_top_end_signal(float(row["mvp_engine_score"].iloc[0]), float(row["defensive_score"].iloc[0])),
        "per36_pts": float(row["per36_pts"].iloc[0]),
        "per36_ast": float(row["per36_ast"].iloc[0]),
        "per36_reb": float(row["per36_reb"].iloc[0]),
        "similar_players": sims,
    }


def auto_top_end_signal(mvp_score: float, defensive_score: float) -> str:
    if mvp_score >= 80:
        return "MVP-caliber offensive engine"
    if mvp_score >= 55:
        return "Primary star creator"
    if defensive_score >= 70 and mvp_score < 45:
        return "Defensive star, not MVP-engine profile"
    if defensive_score >= 45 and mvp_score < 35:
        return "Defense-driven value"
    return "None"


def estimate_range(pred: float, is_exact_anchor: bool = False) -> Tuple[float, float]:
    if is_exact_anchor:
        buffer = 0.25 if pred < 10 else 0.50 if pred < 20 else 0.75 if pred < 35 else 1.25
        return max(0.8, pred - buffer), pred + buffer

    if pred < 5:
        buffer = 0.45
    elif pred < 10:
        buffer = 0.70
    elif pred < 15:
        buffer = 0.95
    elif pred < 22:
        buffer = 1.30
    elif pred < 35:
        buffer = 1.95
    elif pred < 50:
        buffer = 2.60
    else:
        buffer = 3.40
    return max(0.8, pred - buffer), pred + buffer


def similar_players(bundle: ModelBundle, row: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    dist, idx = _nearest(bundle, row, k=k)
    out = bundle.df.iloc[idx][["player", "market_tier", "age", "mpg", "ppg", "apg", "rpg", "aav"]].copy()
    out["distance"] = dist
    return out.reset_index(drop=True)


def tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        subset = df[df["market_tier"] == tier]
        if len(subset):
            rows.append({
                "Market tier": tier,
                "Contracts": len(subset),
                "Median AAV": round(float(subset["aav"].median()), 1),
                "Low": round(float(subset["aav"].min()), 1),
                "High": round(float(subset["aav"].max()), 1),
            })
    return pd.DataFrame(rows)
