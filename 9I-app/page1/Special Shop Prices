import streamlit as st
import pandas as pd

st.title("Special Shop Prices")
st.caption("Work In Progress")
st.caption("Disclaimer: Based off past shops, not all items will be in the shop at one time")

shop_data = {
    "Item": [
        "Grade Increase (GI)", "Diamond Skill Trainer (DST)", "Diamond Trainer (DT)",
        "Legendary Skill Change Ticket (LSCT)", "DT Position Change",
        "Special Sig Pack", "Signature Pack", "DST Gold Skill Change", "Legend BSAT",
        "TSD (P/B)", "OVR Amp", "Blue", "Ultimate Trainer", "TSD Position Selective",
        "Team Selective Prime", "USCT", "GI Reset", "Ult Vintage Pack", "Player Upgrade Ticket",
        "Green", "Diamond BSAT", "Premium Skill Trainer", "Silver BSAT", "PSCT",
        "Diamond ST EXP", "Intermediate GI", "Premium Trainer", "BD Pieces",
        "League Balls", "FA Ticket"
    ],
    "Star Cost": [
        10000, 7000, 6000, 6000, 3000,
        3000, 3000, 3000, 3000,
        3000, 3000, 3000, 3000, 3000,
        3000, 2500, 2500, 2000, 2000,
        2000, 2000, 1500, 1500, 1400,
        1200, 1000, 1000, 150,
        100, 50
    ],
    "Quantity": [
        1, 1, 1, 1, 1,
        1, 2, 1, 1,
        2, 3, 2, 2, 2,
        2, 1, 3, 3, 3,
        2, 3, 5, 2, 2,
        20, 3, 2, 30,
        10, 50
    ]
}

df_shop = pd.DataFrame(shop_data)
st.table(df_shop)
