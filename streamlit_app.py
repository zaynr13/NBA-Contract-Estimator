from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from model import TIERS, build_model, load_data, predict_aav, tier_summary

st.set_page_config(page_title="NBA Contract Estimator", page_icon="🏀", layout="wide")

@st.cache_resource
def get_bundle():
    df = load_data()
    return build_model(df)

bundle = get_bundle()
df = bundle.df

st.title("NBA Contract Estimator")
st.caption("Created by Zayn Remtulla")
st.write(
    "Estimate NBA contract value from the season-before-signing profile: box-score production, efficiency, minutes, defense, awards, availability, and market comparisons."
)

# Synchronized slider + number input.
def _commit_slider(base_key):
    st.session_state[base_key] = st.session_state[f"{base_key}_slider"]
    st.session_state[f"{base_key}_num"] = st.session_state[f"{base_key}_slider"]


def _commit_number(base_key):
    st.session_state[base_key] = st.session_state[f"{base_key}_num"]
    st.session_state[f"{base_key}_slider"] = st.session_state[f"{base_key}_num"]


def stat_input(label, min_value, max_value, default, step, fmt=None, help_text=None, key=None):
    base = key or label.lower().replace(" ", "_").replace("%", "pct")
    slider_key = f"{base}_slider"
    num_key = f"{base}_num"

    if base not in st.session_state:
        st.session_state[base] = default
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[base]
    if num_key not in st.session_state:
        st.session_state[num_key] = st.session_state[base]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_key,
            help=help_text,
            on_change=_commit_slider,
            args=(base,),
        )
    with c2:
        st.number_input(
            " ",
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=fmt,
            key=num_key,
            label_visibility="collapsed",
            on_change=_commit_number,
            args=(base,),
        )
    return st.session_state[base]

med = df.median(numeric_only=True)

est_tab, comps_tab, check_tab, data_tab = st.tabs(["Estimator", "Similar Players", "Model Check", "Training Data"])

with est_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Player Input")
        player_name = st.text_input("Player name", value="Future Player")
        user_tier = st.selectbox(
            "Market tier",
            TIERS,
            index=TIERS.index("Role"),
            help="Display/range context only. It does not train or control the estimate.",
        )

        cA, cB = st.columns(2)
        with cA:
            age = stat_input("Age", 18, 40, int(round(med.get("age", 26))), 1, "%d", key="age")
            gp = stat_input("Games played", 0, 82, int(round(med.get("gp", 65))), 1, "%d", key="gp")
            mpg = stat_input("Minutes per game", 1.0, 40.0, round(float(med.get("mpg", 25)), 1), 0.1, "%.1f", key="mpg")
            ppg = stat_input("Points per game", 0.0, 40.0, round(float(med.get("ppg", 12)), 1), 0.1, "%.1f", key="ppg")
            apg = stat_input("Assists per game", 0.0, 15.0, round(float(med.get("apg", 2)), 1), 0.1, "%.1f", key="apg")
            rpg = stat_input("Rebounds per game", 0.0, 20.0, round(float(med.get("rpg", 4)), 1), 0.1, "%.1f", key="rpg")
            ts_pct = stat_input("True shooting %", 0.400, 0.850, round(float(med.get("ts_pct", 0.58)), 3), 0.001, "%.3f", key="ts")
        with cB:
            per = stat_input("PER", 5.0, 35.0, round(float(med.get("per", 14)), 1), 0.1, "%.1f", key="per")
            win_shares = stat_input("Win shares", 0.0, 20.0, round(float(med.get("win_shares", 3)), 1), 0.1, "%.1f", key="ws")
            bpm = stat_input("BPM", -6.0, 15.0, round(float(med.get("bpm", 0)), 1), 0.1, "%.1f", key="bpm")
            vorp = stat_input("VORP", -2.0, 10.0, round(float(med.get("vorp", 0.7)), 1), 0.1, "%.1f", key="vorp")
            team_wins = stat_input("Team wins", 0, 82, int(round(med.get("team_wins", 42))), 1, "%d", key="wins")
            all_stars = stat_input("All-Star selections", 0, 15, int(round(med.get("all_stars", 0))), 1, "%d", key="allstars")
            popularity_score = stat_input(
                "Popularity score",
                0,
                100,
                int(round(med.get("popularity_score", 60))),
                1,
                "%d",
                help_text="0-100 marketability proxy. Keep consistent across players.",
                key="pop",
            )

        st.markdown("#### Awards, availability, defense")
        d1, d2 = st.columns(2)
        with d1:
            all_nba = stat_input("Career All-NBA selections", 0, 10, int(round(med.get("all_nba", 0))), 1, "%d", key="allnba")
            all_defense = stat_input("Career All-Defense selections", 0, 10, int(round(med.get("all_defense", 0))), 1, "%d", key="alldef")
            availability = stat_input(
                "Availability score",
                0,
                100,
                int(round(med.get("availability", 80))),
                1,
                "%d",
                help_text="Usually GP ÷ 82 × 100, or multi-year availability if you have it.",
                key="avail",
            )
            spg = stat_input("Steals per game", 0.0, 3.0, round(float(med.get("spg", 0.8)), 1), 0.1, "%.1f", key="spg")
            bpg = stat_input("Blocks per game", 0.0, 4.0, round(float(med.get("bpg", 0.4)), 1), 0.1, "%.1f", key="bpg")
        with d2:
            dws = stat_input("Defensive win shares", 0.0, 8.0, round(float(med.get("dws", 2.0)), 1), 0.1, "%.1f", key="dws")
            def_rating = stat_input("Defensive rating", 95, 125, int(round(med.get("def_rating", 115))), 1, "%d", help_text="Lower is better.", key="defrtg")
            dbpm = stat_input("DBPM", -4.0, 6.0, round(float(med.get("dbpm", 0.0)), 1), 0.1, "%.1f", key="dbpm")

    values = {
        "age": age,
        "gp": gp,
        "mpg": mpg,
        "ppg": ppg,
        "apg": apg,
        "rpg": rpg,
        "ts_pct": ts_pct,
        "per": per,
        "win_shares": win_shares,
        "bpm": bpm,
        "vorp": vorp,
        "team_wins": team_wins,
        "all_stars": all_stars,
        "popularity_score": popularity_score,
        "all_nba": all_nba,
        "all_defense": all_defense,
        "availability": availability,
        "spg": spg,
        "bpg": bpg,
        "dws": dws,
        "def_rating": def_rating,
        "dbpm": dbpm,
    }
    result = predict_aav(bundle, values)

    with right:
        st.subheader("Contract Estimate")
        st.markdown(f"### {player_name}")
        st.info(f"Market tier selected: **{user_tier}** *(display only)*")

        if result["is_exact_anchor"]:
            st.caption("This input lands exactly on the trained market surface, so the model-fit estimate equals the actual training AAV.")
        else:
            st.caption("Future estimate uses the trained market surface: real similar contracts + smooth regression. Market tier does not force the number.")

        st.metric("Estimated fair annual value", f"${result['estimate']:.1f}M / year")
        st.markdown(f"**Suggested range:** ${result['low']:.1f}M – ${result['high']:.1f}M")
        st.markdown(f"**Model-implied tier:** {result['implied_tier']}")
        st.markdown(f"**Raw regression estimate:** ${result['raw_regression_estimate']:.1f}M")
        if result.get("adjusted_regression_estimate") != result.get("raw_regression_estimate"):
            st.markdown(f"**Adjusted regression estimate:** ${result['adjusted_regression_estimate']:.1f}M")
            for note in result.get("adjustment_notes", []):
                st.caption(note)
        st.markdown(f"**Local market estimate:** ${result['local_market_estimate']:.1f}M")
        st.caption(f"Dynamic local weight: {result['surface_weight']:.2f} | Similar-contract spread: ${result['local_spread']:.1f}M | Nearest distance: {result['nearest_distance']:.3f}")
        st.caption(f"Local reliability: {result['local_reliability']}")
        st.caption("Raw regression estimates from the full stats pattern; local market compares the player to nearby real contracts.")
        st.divider()
        st.markdown(f"**Defensive score:** {result['defensive_score']:.1f}")
        st.markdown(f"**MVP engine score:** {result['mvp_engine_score']:.1f}")
        st.markdown(f"**Specialist shooter score:** {result['specialist_shooter_score']:.1f}")
        st.markdown(f"**Auto top-end signal:** {result['auto_top_end_signal']}")
        st.markdown(f"**Per-36 scoring:** {result['per36_pts']:.2f} PTS")
        st.markdown(f"**Per-36 assists/rebounds:** {result['per36_ast']:.2f} AST / {result['per36_reb']:.2f} REB")
        if user_tier != result["implied_tier"]:
            st.warning(
                "Selected market tier and model-implied tier differ. The estimate stays stats-first and does not force the number toward the selected tier."
            )

with comps_tab:
    st.subheader("Similar Players")
    st.write("These explain the market-surface estimate. Similar contracts influence the estimate by distance, but market tier still does not force the number.")
    sims = result["similar_players"].copy()
    sims["aav"] = sims["aav"].map(lambda x: round(float(x), 1))
    sims["distance"] = sims["distance"].map(lambda x: round(float(x), 2))
    st.dataframe(sims, use_container_width=True, hide_index=True)

with check_tab:
    st.subheader("Model Check")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training contracts", len(df))
    c2.metric("Market-surface fit MAE", f"${bundle.train_mae:.2f}M")
    c3.metric("Raw regression diagnostic MAE", f"${bundle.cv_mae:.2f}M")

    st.write("The market-surface fit checks how closely the fitted contract surface matches the training contracts.")
    st.write("Raw regression diagnostics are shown only to reveal smoothing bias, not as the main estimate users receive.")

    check = bundle.residual_table.copy()
    st.markdown("#### Actual vs Raw Regression Estimate")
    plot_df = check.copy()
    plot_df["diff"] = plot_df["aav"] - plot_df["raw_regression_estimate"]
    max_val = float(max(plot_df["aav"].max(), plot_df["raw_regression_estimate"].max()) + 5)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["raw_regression_estimate"],
            y=plot_df["aav"],
            mode="markers",
            name="Players",
            marker=dict(size=8, color="rgba(40,40,40,0.75)"),
            customdata=plot_df[["player", "market_tier", "raw_regression_estimate", "aav", "raw_regression_error"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Market tier: %{customdata[1]}<br>"
                "Regression estimate: $%{customdata[2]:.1f}M<br>"
                "Actual AAV: $%{customdata[3]:.1f}M<br>"
                "Error: %{customdata[4]:+.1f}M<extra></extra>"
            ),
        )
    )
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Perfect match", line=dict(color="#4c78ff", width=3)))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[5, max_val + 5], mode="lines", name="Actual +$5M", line=dict(color="#e45756", width=2)))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[-5, max_val - 5], mode="lines", name="Actual -$5M", line=dict(color="#4daf6b", width=2)))
    fig.update_layout(
        xaxis_title="Raw Regression Estimate (Million per year)",
        yaxis_title="Actual AAV (Million per year)",
        height=560,
        hovermode="closest",
        legend_title_text="",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[-5, max_val + 5])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover over any point to see the player, raw regression estimate, actual AAV, and error.")

    check["abs_raw_error"] = (check["future_sim_estimate"] - check["aav"]).abs()
    check = check.sort_values("abs_raw_error", ascending=False)
    display = check.rename(
        columns={
            "player": "Player",
            "market_tier": "Market tier",
            "aav": "Actual AAV",
            "market_surface_fit_estimate": "Market-surface fit estimate",
            "market_surface_fit_error": "Market-surface fit error",
            "raw_regression_estimate": "Raw regression estimate",
            "raw_regression_error": "Raw regression error",
            "future_sim_estimate": "Raw regression diagnostic estimate",
            "future_sim_error": "Raw regression diagnostic error",
            "abs_raw_error": "Abs raw diagnostic error",
        }
    )
    money_cols = [
        "Actual AAV",
        "Market-surface fit estimate",
        "Market-surface fit error",
        "Raw regression estimate",
        "Raw regression error",
        "Raw regression diagnostic estimate",
        "Raw regression diagnostic error",
        "Abs raw diagnostic error",
    ]
    for col in money_cols:
        display[col] = display[col].map(lambda x: round(float(x), 1))

    st.markdown("#### Market-surface fit + raw regression diagnostic")
    st.caption(
        "Use Market-surface fit to verify that training contracts align with actual AAV. Raw regression is only shown to diagnose smoothing bias."
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("#### Tier summary")
    st.dataframe(tier_summary(df), use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Training Data")
    st.write("Training stats should come from the season before the contract was signed.")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download training CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="nba_contract_training_data.csv",
        mime="text/csv",
    )
