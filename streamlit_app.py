import numpy as np
import streamlit as st

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

    tab1, tab2= st.tabs(["Home", "5T-Calculator"])

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

if __name__ == "__main__":
    main()
