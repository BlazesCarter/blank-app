import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def train_needed_5T(base_stats, total_points, stat_labels, bsat_adjustment=None):
    """
    Calculate the highest cap achievable, the lowest target bonus train\
    that provides the same bonus, and the difficulty.

    Args:
    base_stats (list of int): List of base stats.
    total_points (int): Total points available to allocate.
    stat_labels (list of str): Labels for the stats.
    bsat_adjustment (tuple, optional): Indices and amount for BSAT adjustment.

    Returns:
    str: The formatted output string showing the highest cap, associated \
         bonus, the train for the bonus, and any remaining extra points, \
         along with difficulty. If BSAT is used, also shows adjusted train.
    """
    # Define bonus tiers
    bonus_tiers = [
        (72, 80, 8), (81, 82, 8), (83, 84, 9), (85, 87, 10), 
        (88, 89, 11), (90, 92, 12), (93, 94, 13), (95, 97, 14), 
        (98, 99, 15), (100, 102, 16), (103, 104, 17), (105, 107, 18), 
        (108, 109, 19), (110, 112, 20), (113, 114, 21), (115, 117, 22), 
        (118, 119, 23), (120, 120, 24)
    ]

    def get_bonus(stat):
        for low, high, bonus in bonus_tiers:
            if low <= stat <= high:
                return bonus
        return 0

    def calculate_train(stats, total_points):
        """
        Calculate the train distribution and associated metrics based on stats and available points.

        Args:
        stats (list of int): Starting stats (base stats or adjusted stats).
        total_points (int): Total points available to allocate.

        Returns:
        tuple: Train distribution, cap, cap bonus, target bonus value, remaining points, difficulty, interpretation.
        """
        train = [0] * len(stats)
        updated_stats = stats[:]
        sorted_indices = sorted(range(len(updated_stats)), key=lambda x: updated_stats[x])

        # Distribute points to the lowest stats
        for _ in range(total_points):
            updated_stats[sorted_indices[0]] += 1
            train[sorted_indices[0]] += 1
            sorted_indices = sorted(sorted_indices, key=lambda x: updated_stats[x])

        # Calculate the cap and associated bonus
        cap = min(updated_stats)
        cap_bonus = get_bonus(cap)

        # Find the lowest stat value (target bonus value) that provides the same bonus
        target_bonus_value = cap
        for target in range(cap, 71, -1):
            if get_bonus(target) == cap_bonus:
                target_bonus_value = target
            else:
                break

        # Redistribute points to meet the target bonus value
        remaining_points = total_points
        updated_stats = stats[:]
        train = [0] * len(stats)
        sorted_indices = sorted(range(len(updated_stats)), key=lambda x: updated_stats[x])

        for i in sorted_indices:
            to_add = max(0, target_bonus_value - updated_stats[i])
            if remaining_points >= to_add:
                updated_stats[i] += to_add
                train[i] += to_add
                remaining_points -= to_add
            else:
                updated_stats[i] += remaining_points
                train[i] += remaining_points
                remaining_points = 0

        # Calculate the standard deviation of the train distribution for difficulty
        train_std = np.std(train)
        difficulty = min(10, max(0, train_std / (total_points / len(base_stats)) * 10))

        if difficulty >= 9:
            difficulty_interpretation = "Highly statistically unlikely"
        elif difficulty >= 8:
            difficulty_interpretation = "Statistically unlikely"
        elif difficulty >= 6:
            difficulty_interpretation = "Moderately difficult"
        elif difficulty >= 4:
            difficulty_interpretation = "Somewhat difficult"
        else:
            difficulty_interpretation = "Easy"

        return train, cap, cap_bonus, target_bonus_value, remaining_points, difficulty, difficulty_interpretation
    
    # Calculate the original train distribution
    original_train, cap, cap_bonus, target_bonus_value, remaining_points, difficulty, difficulty_interpretation = calculate_train(base_stats[:], total_points)
    
    result = (
        f"Cap {cap} - Bonus {target_bonus_value}; +{cap_bonus} at lvl 8\n"
        f"Train for bonus:\n{' '.join(map(str, original_train))} with extra {remaining_points} to go wherever you want\n"
        f"Difficulty: {difficulty:.2f}/10 - {difficulty_interpretation}\n"
    )
    
    # Handle BSAT adjustment
    if bsat_adjustment:
        subtract_idx, add_idx, bsat_amount = bsat_adjustment
        bsat_stats = base_stats[:]

        # Apply BSAT adjustments to the stats
        bsat_stats[subtract_idx] = max(1, bsat_stats[subtract_idx] - bsat_amount)  # Ensure stats don't go below 1
        bsat_stats[add_idx] = min(120, bsat_stats[add_idx] + bsat_amount)  # Ensure stats don't exceed 120

        # Recalculate the train distribution and difficulty with the adjusted stats
        adjusted_train, cap, cap_bonus, target_bonus_value, remaining_points, difficulty, difficulty_interpretation = calculate_train(bsat_stats[:], total_points)

        # Append the adjusted results to the output
        result += (
            f"\nAfter BSAT adjustment (-{bsat_amount} {stat_labels[subtract_idx]} +{bsat_amount} {stat_labels[add_idx]}):\n"
            f"Cap {cap} - Bonus {target_bonus_value}; +{cap_bonus} at lvl 8\n"
            f"Train for bonus:\n{' '.join(map(str, adjusted_train))} with extra {remaining_points} to go wherever you want\n"
            f"Difficulty: {difficulty:.2f}/10 - {difficulty_interpretation}\n"
        )
    
    return result

def main():
    st.title("9-Innings Tools")

    tab1, tab2, tab3 = st.tabs(["Home", "5T-Calculator","Team Sig Odds"])

    with tab1:
        st.header("Welcome")
        st.write("This is a work in progress, let me know what to add or fix.")

    with tab2:
        st.header("5T Calculator")
        st.write("All values are the original base + GI")

        stat_labels = ["CON", "POW", "EYE", "SPD", "FLD"]
        base_stats = [st.number_input(f"Enter {label}:", min_value=1, max_value=150, value=90) for label in stat_labels]

        supreme = st.radio("Supreme?", ("No", "Yes"))
        total_points = 87 if supreme == "Yes" else 57

        use_bsat = st.checkbox("Try with BSAT")
        bsat_adjustment = None
        if use_bsat:
            subtract_stat = st.selectbox("Stat to decrease:", stat_labels, index=1)
            add_stat = st.selectbox("Stat to increase:", stat_labels, index=4)
            bsat_amount = st.slider("Amount to adjust:", 1, 5, 1)
            if subtract_stat != add_stat:
                bsat_adjustment = (stat_labels.index(subtract_stat), stat_labels.index(add_stat), bsat_amount)

        if st.button("Calculate", key="calculate_train"):
            result = train_needed_5T(base_stats, total_points, stat_labels, bsat_adjustment)
            st.text_area("Results:", result, height=200)
        # Tab 3: Team Signature Odds
    with tab3:
        st.header("Team Signature Odds")

        # Team signature counts
        team_sigs = {
            "Arizona Diamondbacks": 66, "Atlanta Braves": 72, "Baltimore Orioles": 59,
            "Boston Red Sox": 76, "Chicago Cubs": 71, "Chicago White Sox": 62, "Cincinnati Reds": 67,
            "Cleveland Guardians": 67, "Colorado Rockies": 65, "Detroit Tigers": 58, "Houston Astros": 78,
            "Kansas City Royals": 56, "Los Angeles Angels": 59, "Los Angeles Dodgers": 90,
            "Miami Marlins": 63, "Milwaukee Brewers": 58, "Minnesota Twins": 69, "New York Mets": 69,
            "New York Yankees": 75, "Oakland Athletics": 64, "Philadelphia Phillies": 70,
            "Pittsburgh Pirates": 67, "San Diego Padres": 64, "San Francisco Giants": 56,
            "Seattle Mariners": 59, "St. Louis Cardinals": 63, "Tampa Bay Rays": 51,
            "Texas Rangers": 61, "Toronto Blue Jays": 64, "Washington Nationals": 69
        }

        # Team abbreviations
        team_abbr = {
            "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
            "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
            "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
            "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
            "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH"
        }

        # Calculate odds and packs
        total_sigs = sum(team_sigs.values())
        team_odds = {team: count / total_sigs * 100 for team, count in team_sigs.items()}
        team_packs = {team: round(total_sigs / count, 2) if count > 0 else "N/A" for team, count in team_sigs.items()}

        # Dropdown for team selection
        selected_team = st.selectbox("Select a Team:", list(team_sigs.keys()))

        # Display selected team stats
        if selected_team:
            st.subheader(f"Stats for {selected_team}:")
            st.write(f"**Total Sigs:** {team_sigs[selected_team]}")
            st.write(f"**Odds to Pull a Sig:** {team_odds[selected_team]:.2f}%")
            st.write(f"**Sig Packs/Combos Needed (on average):** {team_packs[selected_team]}")

        # Sorting options
        sort_order = st.radio("Sort by:", ["Descending", "Ascending"])
        sorted_teams = sorted(team_odds.items(), key=lambda x: x[1], reverse=(sort_order == "Descending"))
        df = pd.DataFrame(sorted_teams, columns=["Team", "Odds (%)"])

        # Get min and max values for dynamic scaling
        min_value = df["Odds (%)"].min()
        max_value = df["Odds (%)"].max()
        scale_min = min_value - 0.25  # 0.5 lower than min
        scale_max = max_value + 0.25  # 0.5 higher than max

        # Checkbox for dynamic axis scaling
        dynamic_scaling = st.checkbox("Enable Dynamic Axis Scaling")

        # Create an interactive bar chart with dynamic scaling
        fig = go.Figure()

        # Add bar chart
        fig.add_trace(go.Bar(
            x=[team_abbr[team] for team in df["Team"]],
            y=df["Odds (%)"],
            text=[f"{val:.2f}%" for val in df["Odds (%)"]],  # Format text to 2 decimal places
            textposition='outside',
            name="Odds (%)"
        ))

        # Apply dynamic scaling if enabled
        if dynamic_scaling:
            fig.update_layout(yaxis=dict(range=[scale_min, scale_max]))

        # Update layout for better appearance
        fig.update_layout(
            title="Team Signature Odds",
            xaxis_title="Team",
            yaxis_title="Odds (%)",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_tickangle=-45  # Tilt text up and to the right
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # Add a scrollable table for detailed odds
        with st.expander("View Detailed Odds Table"):
            st.dataframe(df, height=300)  # Scrollable table

        # Add a download button for the data
        csv = df.to_csv(index=False)
        st.download_button(label="Download Data as CSV", data=csv, file_name="team_signature_odds.csv", mime="text/csv")

if __name__ == "__main__":
    main()
