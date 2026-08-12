# Product_Recommendation_System
This is Machine Learning &amp; Natural language processing project that recommend similar to product selected by user.


## Dataset
Dataset link = https://www.kaggle.com/datasets/asaniczka/azon-products-dataset-2023-1-4m-products

Using a amazon_products.csv dataset from keggle.

## Website

website link :- https://product-recommendation-system-333r.onrender.com/

## What We Did

- Prepared data as a dataframe performing data cleaning process
- Extract Useful coloumn for this project like "title, asin, producturl, categories_id
- Generated vectors of title coloumn data
- Generated cosine_similarity for all title respect with all product and store it.
- Make a function that take user input and then suggest recommendation.
- Push code to github and deploy on render.

## Libraries Used

pandas | sklearn | nltk | scipy | joblib | streamlit

## How to Run

```
Create a virtual environment
Activate the virtual environment
Install required libraries:
    streamlit
    scikit-learn
    joblib
    scipy    
    pandas
    nltk

Run the Python script or Jupyter notebook
```

## Project Structure

GENAI-TASK(20)-AMAN/
│── Task.ipynb
│── requirements.txt
|── amanzon_products.csv
|── app.py
|── .gitattributes
|── tfidf_matrix.npz
|── tfidf_vectorizer.pkl
│── README.md
```

---

## Conclusion

This project demonstrates the fundamentals of **Word2Vec embeddings**, including **CBOW, Skip-Gram, word similarity, vector operations, and embedding visualization**, providing a practical introduction to word representation learning in NLP.

<!-- Aman Kumar -->