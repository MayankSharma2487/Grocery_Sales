import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from predict import predict_sales_profit



st.set_page_config(page_title="Supermart Sales Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()
    df["Week"] = df["Order Date"].dt.isocalendar().week
    df["Day"] = df["Order Date"].dt.day_name()
    return df

df = load_data()


st.title("🛒 Supermart Sales & Profit Analysis")

# Sidebar filters (dropdown style)
st.sidebar.header("📊 Filters")

regions = sorted(df["Region"].dropna().unique())
region_choice = st.sidebar.selectbox(
    "Select Region",
    options=["All"] + regions,
    index=0,
    key="region_select",
)

    # ----- Year dropdown -----
years = sorted(df["Year"].dropna().astype(int).unique())
year_choice = st.selectbox(
        "Select Year",
        ["All"] + [str(y) for y in years],
        index=0,
        key="year_select"
    )



    # ----- City dropdown (dependent on Region) -----
if region_choice == "All":
        cities_filtered = sorted(df["City"].dropna().unique())
else:
    cities_filtered = sorted(
            df.loc[df["Region"] == region_choice, "City"].dropna().unique()
        )

city_choice = st.selectbox(
        "Select City",
        ["All"] + cities_filtered,
        index=0,
        key="city_select"
    )
# ───────────────────────────────────────────────────────────────────────────


# Apply filters
filtered_df = df.copy()
if region_choice != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region_choice]
if year_choice != "All":
    filtered_df = filtered_df[filtered_df["Year"] == int(year_choice)]
if city_choice != "All":
    filtered_df = filtered_df[filtered_df["City"] == city_choice]




st.markdown("### 🧮 Key Metrics")
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"₹{total_sales:,.2f}")
col2.metric("Total Profit", f"₹{total_profit:,.2f}")
col3.metric("Profit Margin", f"{profit_margin:.2f}%")

if region_choice != "All" and year_choice != "All" and city_choice != "All":
    st.markdown(f"### 🌍 Sales & Profit for {city_choice} City in {region_choice} Region ({year_choice})")
else:
    st.markdown("### 🌍 Sales & Profit by All Regions, Cities, and Years")


col1, col2 = st.columns(2)

with col1:
    region_group = filtered_df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    st.bar_chart(region_group)

with col2:
    state_group = (
        filtered_df.groupby("State")[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    st.bar_chart(state_group)


st.markdown("### 🏷️ Category Performance")

tab1, tab2 = st.tabs(["Category", "Sub Category"])

with tab1:
    cat_group = filtered_df.groupby("Category")[["Sales", "Profit"]].sum()
    st.bar_chart(cat_group)

with tab2:
    subcat_group = (
        filtered_df.groupby("Sub Category")[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(15)
    )
    st.bar_chart(subcat_group)


st.markdown("### ⏳ Sales & Profit Over Time")

time_group = (
    filtered_df.groupby("Month")[["Sales", "Profit"]]
    .sum()
    .reindex([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
)

with st.expander("📆 View More Time-Based Trends"):
    tab_week, tab_day = st.tabs(["Weekly", "Day of Week"])
    
    with tab_week:
        week_group = filtered_df.groupby("Week")[["Sales", "Profit"]].sum()
        st.line_chart(week_group)

    with tab_day:
        day_group = filtered_df.groupby("Day")[["Sales", "Profit"]].sum().reindex([
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        st.bar_chart(day_group)

st.markdown("### 🏆 Top 10 Sub-Categories by Sales")
top_products = (
    filtered_df.groupby("Sub Category")[["Sales", "Profit"]]
    .sum()
    .sort_values("Sales", ascending=False)
    .head(10)
    .reset_index()
)
st.dataframe(top_products)

st.markdown("### 📅 Top Selling Sub-Categories by Day of Week")

# Reorder days to match the calendar
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
filtered_df["Day"] = pd.Categorical(filtered_df["Day"], categories=day_order, ordered=True)

# Group by Day and Sub Category
day_subcat = (
    filtered_df.groupby(["Day", "Sub Category"])["Sales"]
    .sum()
    .reset_index()
    .sort_values(["Day", "Sales"], ascending=[True, False])
)

# Get top sub-category per day
top_subcats_by_day = (
    day_subcat.groupby("Day").first().reset_index()
)

st.dataframe(top_subcats_by_day)

monthly_trend = (
    filtered_df.groupby(["Year", "Month"])[["Sales", "Profit"]]
    .sum()
    .reset_index()
)

st.markdown("### 📈 Sales and Profit Trend Over Time")
fig, ax = plt.subplots(figsize=(10, 4))
monthly_trend.plot(kind="line", x="Month", y=["Sales", "Profit"], ax=ax)
st.pyplot(fig)


st.markdown("### 🔮 Predict Sales and Profit")

with st.form("predict_form"):
    category = st.selectbox("Category", df["Category"].unique())
    sub_category = st.selectbox("Sub Category", df["Sub Category"].unique())
    city = st.selectbox("City", df["City"].unique())
    region = st.selectbox("Region", df["Region"].unique())
    discount = st.number_input("Discount", 0.0, 1.0, 0.0, 0.01)
    profit_input = st.number_input("Profit", step=100.0)
    discounted_price = st.number_input("Discounted Price", step=100.0)
    category_sub = category + "_" + sub_category
    city_avg_sales = st.number_input("City Avg Sales", step=100.0)
    city_avg_profit = st.number_input("City Avg Profit", step=100.0)
    order_month = st.number_input("Order Month", 1, 12)
    order_year = st.number_input("Order Year", 2010, 2030)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            "Category": category,
            "Sub Category": sub_category,
            "City": city,
            "Region": region,
            "Discount": discount,
            "Profit": profit_input,
            "Discounted_Price": discounted_price,
            "Category_Sub": category_sub,
            "City_Avg_Sales": city_avg_sales,
            "City_Avg_Profit": city_avg_profit,
            "Order Month": order_month,
            "Order Year": order_year,
        }

        sales, profit = predict_sales_profit(input_data)
        st.success(f"Predicted Sales: ₹{sales:.2f}")
        st.success(f"Predicted Profit: ₹{profit:.2f}")





fig, ax = plt.subplots(figsize=(10, 4))
time_group.plot(kind="bar", ax=ax)
st.pyplot(fig)


# ───────────────────────────────────────────────────────────────────────────
# 🔮 FUTURE FORECAST SECTION
st.markdown("## 🔮 Future Sales & Profit Forecast (Time Series Based)")

# Group monthly sales and profit (already cleaned)
forecast_df = (
    df.groupby(pd.Grouper(key="Order Date", freq="M"))[["Sales", "Profit"]]
    .sum()
    .reset_index()
)

# Prepare data
forecast_df["Month_Year"] = forecast_df["Order Date"].dt.to_period("M").astype(str)
forecast_df["Timestamp"] = forecast_df["Order Date"].map(pd.Timestamp.toordinal)

from sklearn.linear_model import LinearRegression

# Fit models
X = forecast_df[["Timestamp"]]
sales_model = LinearRegression().fit(X, forecast_df["Sales"])
profit_model = LinearRegression().fit(X, forecast_df["Profit"])

# Forecast next 12 months
future_months = pd.date_range(start=forecast_df["Order Date"].max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
future_ordinals = future_months.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

future_sales = sales_model.predict(future_ordinals)
future_profit = profit_model.predict(future_ordinals)

# Create forecast DataFrame
forecast_output = pd.DataFrame({
    "Month": future_months,
    "Predicted Sales": future_sales,
    "Predicted Profit": future_profit
})

# Plotting it 🔥
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(forecast_output["Month"], forecast_output["Predicted Sales"], label="Predicted Sales", marker='o')
ax.plot(forecast_output["Month"], forecast_output["Predicted Profit"], label="Predicted Profit", marker='x')
ax.set_title("📆 Forecast: Sales & Profit for Next 12 Months")
ax.set_xlabel("Month")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Optional: show the data
with st.expander("📄 View Forecast Table"):
    st.dataframe(forecast_output)
