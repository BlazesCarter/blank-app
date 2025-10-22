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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "5T-Calculator","Team Sig Odds","FIN/DOM Calculator", "Donation Event Calculator"])

    with tab1:
        st.header("Welcome")
        st.write("This site is not in any way affiliated with Com2Us, nor is it acting as or claiming to be an official resource.")

    with tab2:
        st.header("5T Calculator")
        st.markdown("All values are the Original Base + GI\
                      \nIf you already are using a BSAT note that below are the Original Bases + GI\n\
           \n**Difficulty Rating Explanation:**\
           \n- 0-4: Easy\
           \n- 4-6: Somewhat difficult\
           \n- 6-8: Moderately difficult\
           \n- 8-9: Statistically unlikely\
           \n- 9-10: Highly statistically unlikely")

        stat_labels = ["CON", "POW", "EYE", "SPD", "FLD"]
        cols = st.columns(5)
        base_stats = [cols[i].number_input(f"{stat_labels[i]}", min_value=1, max_value=150, value=90) for i in range(5)]

        supreme = st.radio("Supreme?", ("No", "Yes"))
        total_points = 87 if supreme == "Yes" else 57

        use_bsat = st.checkbox("Try with BSAT")
        show_thresholds = st.checkbox("Show 5-Tool Boost Thresholds")

        if show_thresholds:
            boost_data = {
                "Lowest Stat (Base+GI+Dev)": [
                    "72–80", 81, 82, 83, 84, 85, 86, 87, 88, 89,
                    90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 101, 102, 103, 104, 105, 106
                ],
                "Stat Increase (Lv. 7)": [
                    "+7", "+7", "+7", "+8", "+8", "+8", "+9", "+9", "+9", "+10",
                    "+10", "+10", "+10", "+11", "+11", "+12", "+12", "+12", "+13", "+13",
                    "+14", "+14", "+14", "+15", "+15", "+16", "+16"
                ],
                "Stat Increase (Lv. 8)": [
                    "+8", "+8", "+8", "+9", "+9", "+10", "+10", "+10", "+11", "+11",
                    "+12", "+12", "+12", "+13", "+13", "+14", "+14", "+14", "+15", "+15",
                    "+16", "+16", "+16", "+17", "+17", "+18", "+18"
                ]
            }
            df_boost = pd.DataFrame(boost_data)
            st.table(df_boost)
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
        st.write("Updated as of 2025 8th Live Update")

        # Team signature counts
        team_sigs = {
            "Arizona Diamondbacks": 56, "Atlanta Braves": 67, "Baltimore Orioles": 59,
            "Boston Red Sox": 75, "Chicago Cubs": 67, "Chicago White Sox": 52,
            "Cincinnati Reds": 62, "Cleveland Guardians": 75, "Colorado Rockies": 56,
            "Detroit Tigers": 57, "Houston Astros": 74, "Kansas City Royals": 50,
            "Los Angeles Angels": 58, "Los Angeles Dodgers": 89, "Miami Marlins": 54,
            "Milwaukee Brewers": 56, "Minnesota Twins": 65, "New York Mets": 67,
            "New York Yankees": 72, "Oakland Athletics": 60, "Philadelphia Phillies": 68,
            "Pittsburgh Pirates": 72, "San Diego Padres": 58, "San Francisco Giants": 54,
            "Seattle Mariners": 52, "St. Louis Cardinals": 60, "Tampa Bay Rays": 47,
            "Texas Rangers": 55, "Toronto Blue Jays": 63, "Washington Nationals": 68
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
            st.subheader(f"{selected_team}:")
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
            textfont=dict(size=25),
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
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=15)
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # Add a scrollable table for detailed odds
        with st.expander("View Detailed Odds Table"):
            st.dataframe(df, height=300)  # Scrollable table

        # Add a download button for the data
        csv = df.to_csv(index=False)
        st.download_button(label="Download Data as CSV", data=csv, file_name="team_signature_odds.csv", mime="text/csv")

    with tab4:
            st.header("FIN/DOM Calculator")
    
            st.markdown("""
            **Directions:**
            - Enter your pitcher's stats below.
            - **To optimize a FIN train**: **LOC + BRK** must be **5+ higher** than **VEL + FB**
            - **To optimize a DOM train**: **VEL + FB** must be **5+ higher** than **LOC + BRK**
    
            - Disclaimer: Gear will affect FIN/DOM Bonuses
            """)
    
            stat_labels = ["LOC", "VEL", "STA", "FB", "BRK"]
    
            # === Split input into Base, GI, and Development (Train)
            st.subheader("Base Stats")
            base_cols = st.columns(5)
            base_stats = [base_cols[i].number_input(f"{stat_labels[i]}", min_value=0, max_value=150, value=50, key=f"base_{i}") for i in range(5)]
    
            st.subheader("Grade Increase (GI)")
            gi_cols = st.columns(5)
            gi_stats = [gi_cols[i].number_input(f"{stat_labels[i]}", min_value=0, max_value=90, value=15, key=f"gi_{i}") for i in range(5)]
    
            st.subheader("Development (Train)")
            train_cols = st.columns(5)
            train_stats = [train_cols[i].number_input(f"{stat_labels[i]}", min_value=0, max_value=87, value=0, key=f"train_{i}") for i in range(5)]
    
            st.subheader("Amp Tickets")
            st.caption("(If applicable)")
            amp_cols = st.columns(5)
            amps = [amp_cols[i].number_input(f"{stat_labels[i]}", min_value=0, max_value=15, value=0, key=f"amp_{i}") for i in range(5)]
    
            st.subheader("Special Training Level")
            st_level = st.slider("ST Level", 0, 10, 0)
    
            def apply_special_training(base_stats, gi_stats, train_stats, amps, st_level):
                # GI + Dev (Train)
                train_total = [gi_stats[i] + train_stats[i] for i in range(5)]
                pre_st_total = [base_stats[i] + train_total[i] + amps[i] for i in range(5)]
    
                # Sort indices based on: Train > Distribution > Base > Left-to-right
                sorted_indices = sorted(
                    range(5),
                    key=lambda i: (
                        -train_stats[i],                 # 1. Development only
                        -pre_st_total[i],               # 2. Distribution
                        -base_stats[i],                 # 3. Base stat
                        i                               # 4. Left to right
                    )
                )
    
                st_bonus = [0] * 5
                if st_level >= 1:
                    st_bonus[sorted_indices[0]] += 2
                if st_level >= 2:
                    st_bonus[sorted_indices[1]] += 2
                if st_level >= 3:
                    st_bonus[sorted_indices[0]] += 2
                if st_level >= 4:
                    st_bonus[sorted_indices[1]] += 2
                if st_level >= 5:
                    st_bonus[sorted_indices[0]] += 2
                if st_level >= 7:
                    st_bonus[sorted_indices[0]] += 2
                    st_bonus[sorted_indices[1]] += 2
                if st_level >= 8:
                    st_bonus[sorted_indices[0]] += 2
                    st_bonus[sorted_indices[1]] += 2
                    st_bonus[sorted_indices[2]] += 2
                if st_level >= 9:
                    st_bonus[sorted_indices[0]] += 2
                    st_bonus[sorted_indices[1]] += 2
                    st_bonus[sorted_indices[2]] += 2
    
                final_stats = [base_stats[i] + gi_stats[i] + train_stats[i] + amps[i] + st_bonus[i] for i in range(5)]
                return final_stats
    
            if st.button("Calculate", key="calculate_fin_dom"):
                final_stats = apply_special_training(base_stats, gi_stats, train_stats, amps, st_level)
    
                st.subheader("Final Stats")
                label_cols = st.columns(5)
                for i in range(5):
                    label_cols[i].markdown(f"**{stat_labels[i]}**")
    
                value_cols = st.columns(5)
                for i in range(5):
                    value_cols[i].markdown(f"{final_stats[i]}")
    
                loc_brk = final_stats[0] + final_stats[4]
                vel_fb = final_stats[1] + final_stats[3]
    
                st.markdown("---")
                st.subheader("Result:")
                if loc_brk - vel_fb >= 5:
                    st.success("FIN Lean")
                elif vel_fb - loc_brk >= 5:
                    st.success("DOM Lean")
                else:
                    st.warning("No Lean")
    # ---------------- Tab 5: Donation Event Calculator ----------------
    with tab5:
        st.title("Donation Event Calculator")
        st.caption("(*Based on the 2025 Donation Event points and items)\n"
           "> Player Donation Period: Oct. 13, 1:00 AM - Oct. 26, 10:59 AM EDT\n"
           "> Player Pool Decision Period: Oct. 26, 11:00 AM - Oct. 28, 00:59 AM EDT\n"
           "> Player Draw Period: Oct. 28, 1:00 AM EDT - Nov. 10, 9:59 AM EST\n"
           "> Item Shop Open Period: Oct. 28, 1:00 AM EDT - Nov. 11, 9:59 AM EST")
        st.markdown("### Enter how many of each card tier you have:")

        # --- Card Inputs (Supreme, Legend, Signature, Prime) ---
        # Supreme
        st.subheader("Supreme")
        c1, c2 = st.columns(2)
        with c1:
            supreme_gold = st.number_input("Gold or higher", min_value=0, step=1, key="supreme_gold")
        with c2:
            supreme_silver = st.number_input("Silver", min_value=0, step=1, key="supreme_silver")

        # Legend
        st.subheader("Legend")
        c1, c2 = st.columns(2)
        with c1:
            legend_gold = st.number_input("Gold or higher", min_value=0, step=1, key="legend_gold")
        with c2:
            legend_silver = st.number_input("Silver", min_value=0, step=1, key="legend_silver")

        # Signature
        st.subheader("Signature")
        c1, c2, c3 = st.columns(3)
        with c1:
            signature_diamond = st.number_input("Diamond", min_value=0, step=1, key="signature_diamond")
        with c2:
            signature_gold = st.number_input("Gold", min_value=0, step=1, key="signature_gold")
        with c3:
            signature_silver = st.number_input("Silver", min_value=0, step=1, key="signature_silver")

        # Prime
        st.subheader("Prime (Limited to 50 copies)")
        c1, c2, c3 = st.columns(3)
        with c1:
            prime_diamond = st.number_input("Diamond", min_value=0, step=1, key="prime_diamond")
        with c2:
            prime_gold = st.number_input("Gold", min_value=0, step=1, key="prime_gold")
        with c3:
            prime_silver = st.number_input("Silver", min_value=0, step=1, key="prime_silver")

        # --- PRIME LIMIT CHECK ---
        total_prime = prime_diamond + prime_gold + prime_silver
        if total_prime > 50:
            st.error("You cannot have more than 50 Prime cards total. Please adjust your inputs.")
            total_points = 0
        else:
            # Calculate total points dynamically
            points_map = {
                "Supreme Gold+": 4500,
                "Supreme Silver": 4000,
                "Legend Gold+": 4500,
                "Legend Silver": 4000,
                "Signature Diamond": 1000,
                "Signature Gold": 500,
                "Signature Silver": 300,
                "Prime Diamond": 300,
                "Prime Gold": 150,
                "Prime Silver": 100
            }

            total_points = (
                supreme_gold * points_map["Supreme Gold+"] +
                supreme_silver * points_map["Supreme Silver"] +
                legend_gold * points_map["Legend Gold+"] +
                legend_silver * points_map["Legend Silver"] +
                signature_diamond * points_map["Signature Diamond"] +
                signature_gold * points_map["Signature Gold"] +
                signature_silver * points_map["Signature Silver"] +
                prime_diamond * points_map["Prime Diamond"] +
                prime_gold * points_map["Prime Gold"] +
                prime_silver * points_map["Prime Silver"]
            )
            st.subheader("Total Donation Points")
            st.success(f"**{total_points:,} points**")

        # === ITEM SHOP ===
        st.markdown("### 2025 Item Shop")

        shop_items = [
            {"name": "Team P/B Diamond Pack", "cost": 2000, "limit": 3},
            {"name": "Diamond Trainer", "cost": 5000, "limit": 1},
            {"name": "ALL Historic Box", "cost": 5000, "limit": 1},
            {"name": "Ult Vintage Player Pack", "cost": 4000, "limit": 3},
            {"name": "Special Signature Pack", "cost": 4000, "limit": 2},
            {"name": "Team Selective Sig Pack", "cost": 8000, "limit": 1},
            {"name": "Legend Player Pack", "cost": 10000, "limit": 1},
        ]

        st.markdown("Select how many of each item you want (respects limits and your points):")

        purchase = {}

        # First row: first 4 items
        cols1 = st.columns(4)
        for i in range(4):
            item = shop_items[i]
            with cols1[i]:
                st.markdown(f"{item['cost']:,} pts  \n{item['name']}  \nExchangeable {item['limit']}")
                purchase[item["name"]] = st.number_input("", min_value=0, max_value=item["limit"], step=1, key=item["name"])

        # Second row: last 3 items centered
        cols2 = st.columns([1,2,2,2,1])
        for j in range(3):
            item = shop_items[j + 4]
            with cols2[j + 1]:
                st.markdown(f"{item['cost']:,} pts  \n{item['name']}  \nExchangeable {item['limit']}")
                purchase[item["name"]] = st.number_input("", min_value=0, max_value=item["limit"], step=1, key=item["name"])

        # Calculate total cost and remaining points
        total_cost = sum(purchase[item["name"]] * item["cost"] for item in shop_items)
        remaining_points = total_points - total_cost

        if remaining_points < 0:
            st.warning(f"Not enough points. Over by {-remaining_points:,} points.")
        else:
            st.success(f"Purchases valid. Remaining points: {remaining_points:,}")

        # Purchase summary
        purchase_summary = [
            (item["name"], purchase[item["name"]], item["cost"], purchase[item["name"]] * item["cost"])
            for item in shop_items if purchase[item["name"]] > 0
        ]
        if purchase_summary:
            df_purchase = pd.DataFrame(purchase_summary, columns=["Item", "Quantity", "Cost per Item", "Total Cost"])
            st.subheader("Purchase Summary")
            st.dataframe(df_purchase.style.format({"Cost per Item": "{:,.0f}", "Total Cost": "{:,.0f}"}))
        else:
            st.info("No items selected for purchase.")
if __name__ == "__main__":
        main()
