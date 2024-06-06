import streamlit as st
import pandas as pd
import numpy as np
import json

st.title("Telecom Churn Prediction")

import streamlit as st
import requests

st.text("Please fill in the following details to predict churn")

# Define the initial data
data = dict()
with st.form("prediction_form"):
    data["complains"] = 1 if st.checkbox("Has the user raised any complains") else 0

    c1, c2 = st.columns(2)

    with c1:

        data["call_failure"] = st.number_input(
            "Number of call failures",
            min_value=0,
            max_value=10000,
            value=0,
        )

        data["subscription_length"] = st.number_input(
            "Subscription Length in months",
            min_value=0,
            max_value=1200,
            value=22,
        )
        data["tariff_plan"] = st.selectbox(
            "Tariff plan",
            [0, 1],
            1,
            format_func=lambda x: "Pay as you go" if x == 1 else "Contractual",
        )
        data["age"] = st.number_input(
            "Age",
            min_value=0,
            max_value=123,
            value=35,
        )
        data["charge_amount"] = st.selectbox(
            "Select the payment amount",
            range(0, 10),
            1,
            help="(0 represents the minimum possible payment)",
        )

    with c2:
        data["customer_value"] = st.number_input(
            "Customer value",
            min_value=0.0,
            max_value=2500.0,
            value=830.3,
            help="The calculated value of a customer, ranging from 0 to 2500",
        )
        data["use_per_month"] = st.number_input(
            "Minutes used Per Month",
            min_value=0,
            max_value=6000,
            value=300,
        )
        data["calls_per_month"] = st.number_input(
            "Number of calls made in a month",
            min_value=0,
            max_value=600,
            value=50,
        )
        data["sms_per_month"] = st.number_input(
            "Number of SMS sent in a month",
            min_value=0,
            max_value=300,
            value=31,
        )
        data["dist_nums_per_month"] = st.number_input(
            "Number of distinct numbers called in a month",
            min_value=0,
            max_value=100,
            value=10,
        )

    submitted = st.form_submit_button(
        "Predict!", use_container_width=True, type="primary"
    )
# Now data dictionary contains the values from text inputs

if submitted:
    # Make the API call
    payload = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    response = int(
        requests.post("http://fastapichurn:80/", data=payload, headers=headers).text
    )

    if response == 0:
        st.success("The user is not likely to churn", icon=":material/thumb_up:")
    else:
        st.error("The user is likely to churn", icon=":material/warning:")
