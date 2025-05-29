import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sus
import numpy as np
import matplotlib.pyplot as pit
import yfinance as yf
import json
import os
import warnings

def bubble_sort(seq):
    n = len(seq)
    for i in range(n):
        for j in range(n-1-i):
            if seq[j] > seq[j+1]:
                seq[j], seq[j+1] = seq[j+1], seq[j]
    return seq

def selection_sort(array):
    n = len(array)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if array[j] < array[min_idx]:
                min_idx = j
        array[i], array[min_idx] = array[min_idx], array[i]
    return array

@st.cache_data
def fetch_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def sma_strategy(data, short_window, long_window):
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = 0
    #data.loc[short_window:, 'Signal'] = np.where(data['SMA_Short'][short_window:] > data['SMA_Long'][short_window:], 1, 0)
    data.iloc[short_window:, data.columns.get_loc('Signal')] = np.where(data['SMA_Short'].iloc[short_window:] > data['SMA_Long'].iloc[short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

def plot_signals(data, short, long):
    fig, ax = pit.subplots(figsize=(14, 7))
    ax.plot(data['Close'], label='Close Price', alpha=0.5)
    ax.plot(data['SMA_Short'], label=f'SMA {short}', alpha=0.75)
    ax.plot(data['SMA_Long'], label=f'SMA {long}', alpha=0.75)
    # Plot Buy signals
    ax.plot(data[data['Position'] == 1].index, data['SMA_Short'][data['Position'] == 1], '^', markersize=10, color='green', label='Buy Signal')
    # Plot Sell signals
    ax.plot(data[data['Position'] == -1].index, data['SMA_Short'][data['Position'] == -1], 'v', markersize=10, color='red', label='Sell Signal')
    ax.set_title('SMA Crossover Strategy')
    ax.legend()
    return fig

def plot_checker(size=8):
    board = np.indices((size, size)).sum(axis=0) % 2
    fig, ax = pit.subplots(figsize=(6, 6))
    ax.imshow(board, cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Checker Board Pattern", fontsize=14, fontweight="bold")
    st.pyplot(fig)

def load_books_file(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"failed to upload books file: {str(e)}")
        return None

def applications_starting():
    st.set_page_config(page_title="Books Analysis", page_icon="📚", layout="wide")
    st.subheader("📚 Books Analysis App")

    upload_file = st.file_uploader("Upload Books Record", type=["csv", "xlsx"])
    if upload_file is not None:
        df = load_books_file(upload_file)
    else:
        os.chdir(r"C:\Users\PAUL\Desktop\pythonApplications\books-analysis")
        df = pd.read_csv("books_data.csv")
    
    #st.write(df)
    #st.dataframe(df.head())
    #st.write(df.describe())

    st.sidebar.title("Books Filters")
    genre_filter = st.sidebar.multiselect("Books Genre", df["Genre"].unique())
    year_range = st.sidebar.slider("Published Year Range", int(df["PublishedYear"].min()), int(df["PublishedYear"].max()), (int(df["PublishedYear"].min()), int(df["PublishedYear"].max())) )
    price_range = st.sidebar.slider("Books Price Range", float(df["Price"].min()), float(df["Price"].max()), (float(df["Price"].min()), float(df["Price"].max())) )

    filtered_df = df[df["Genre"].isin(genre_filter) & df["PublishedYear"].between(*year_range) & df["Price"].between(*price_range)]
    #st.dataframe(filtered_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Genre Distribution")
        genre_count = filtered_df["Genre"].value_counts()
        st.bar_chart(genre_count, use_container_width=True)

    with col2:
        st.subheader("Book Published Per Year")
        book_year = filtered_df["PublishedYear"].value_counts().sort_index()
        st.line_chart(book_year, use_container_width=True)

    with col3:
        st.subheader("Genre wise Price")
        genre_df = filtered_df.groupby("Genre")["Price"].sum().reset_index()
        fig = px.pie(genre_df, values="Price", names="Genre", title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Checker Board"):
        plot_checker(8)

    with st.expander("Sorting Out List"):
        listing = [24, 32, 45, 12, 8, 16, 65, 14, 48]
        st.write("List before sort: ", listing)
        sort_list = bubble_sort(listing)
        st.write("List after bubble sort: ", sort_list)
        select_list = selection_sort(listing)
        st.write("List after selection sort: ", select_list)

    with st.expander("plotting histogram"):
        fig, ax = pit.subplots()
        sus.histplot(filtered_df["Price"], bins=10, kde=True, ax=ax)
        st.pyplot(fig, use_container_width=True)

    with st.expander("plotting scatter"):
        st.subheader("📌 Scatterplot: Pages vs. Price")
        fig2, ax2 = pit.subplots()
        sus.scatterplot(data=filtered_df, x="Pages", y="Price", hue="Genre", palette="tab10", ax=ax2)
        ax2.set_title("Pages vs. Price")
        st.pyplot(fig2)

    ticker = st.sidebar.text_input("Ticker Symbol", value='AAPL')
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    short_window = st.sidebar.slider("Short SMA Window", 5, 100, 50)
    long_window = st.sidebar.slider("Long SMA Window", 50, 300, 200)

    data = fetch_data(ticker, start_date, end_date)
    if data.empty:
        st.warning("No data found for the selected ticker and date range.")

    else:
        data = sma_strategy(data, short_window, long_window)
        st.pyplot(plot_signals(data, short_window, long_window))
        st.subheader("📊 Data Preview")
        st.dataframe(data.tail(10))




applications_starting()