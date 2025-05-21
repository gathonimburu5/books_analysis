import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sus
import matplotlib.pyplot as pit
import json
import os
import warnings

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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Genre Distribution")
        genre_count = filtered_df["Genre"].value_counts()
        st.bar_chart(genre_count, use_container_width=True)

    with col2:
        st.subheader("Book Publishe Per Year")
        book_year = filtered_df["PublishedYear"].value_counts().sort_index()
        st.line_chart(book_year, use_container_width=True)
    
    genre_df = filtered_df.groupby("Genre")["Price"].sum().reset_index()
    fig = px.pie(genre_df, values="Price", names="Genre", title="Genre wise Price")
    st.plotly_chart(fig, use_container_width=True)

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


applications_starting()