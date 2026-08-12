from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import streamlit as st
import pandas as pd
import joblib


st.title("Product Recommendation System")

df = pd.read_csv("amazon_products.csv")

vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")

selected_product = st.selectbox("Select a product", options=df["title"].tolist()[0:3000], key="selected_product")


def recommend(item_name, top_n=5):
    if item_name not in df["title"].values:
        return None

    item_index = df[df["title"] == item_name].index[0]

    scores = cosine_similarity(tfidf_matrix[item_index], tfidf_matrix).flatten()

    sorted_items = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sorted_items[1:top_n + 1]]

    return df.loc[top_indices, ["title", "productURL"]]

if st.button("Recommend"):
    result = recommend(selected_product)

    if result is None:
        st.error("Product not found.")
    else:
        st.subheader("Recommended Products")

        for idx, row in enumerate(result.itertuples(index=False), start=1):
            st.markdown(f"**{idx}. {row.title}**")
            st.markdown(f"Product Link: {row.productURL}")