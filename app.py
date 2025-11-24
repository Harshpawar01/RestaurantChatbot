import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.title("Restaurant Menu Recommendation & Chatbot")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('restaurant_menu_1000.csv')
    return df

df = load_data()

# Data Cleaning & Preprocessing
df.drop_duplicates(inplace=True)
df.columns = [col.strip().title() for col in df.columns]
df['Cuisine'] = df['Cuisine'].str.strip().str.title()
if 'Veg/NonVeg' in df.columns:
    df['Veg_NonVeg'] = df['Veg/NonVeg'].map({'Veg':1, 'Non-Veg':0})
df.fillna({'Rating': df['Rating'].mean(), 'Price': df['Price'].median()}, inplace=True)

# Show dataframe sample
if st.checkbox("Show Data Sample"):
    st.write(df.head())

# EDA: Top 10 Cuisines
st.subheader('Top 10 Cuisines')
fig, ax = plt.subplots()
df['Cuisine'].value_counts().head(10).plot(kind='bar', color='skyblue', ax=ax)
st.pyplot(fig)

# EDA: Price vs Rating
st.subheader('Price vs Rating')
fig2, ax2 = plt.subplots()
ax2.scatter(df['Price'], df['Rating'], alpha=0.6)
ax2.set_xlabel('Price')
ax2.set_ylabel('Rating')
ax2.set_title('Price vs Rating')
st.pyplot(fig2)

# Feature Engineering for Recommendation
df['Combined_Features'] = df['Cuisine'].astype(str) + ' ' + df['Dish_Name'].astype(str)
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['Combined_Features'])
similarity = cosine_similarity(count_matrix)

# Dish Recommender Function
def recommend_dishes(dish_name, top_n=5):
    if dish_name not in df['Dish_Name'].values:
        return pd.DataFrame([["Dish not found. Try another name.", "", "", ""]], columns=['Dish_Name','Cuisine','Price','Rating'])
    idx = df[df['Dish_Name'] == dish_name].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended = df.iloc[[i[0] for i in sorted_scores]][['Dish_Name','Cuisine','Price','Rating']]
    return recommended

# Recommendation Section
st.subheader("Get Dish Recommendations")
dish_input = st.text_input("Enter a dish name for recommendations:")
if dish_input:
    st.write(recommend_dishes(dish_input))

# Chatbot Section
st.subheader("Chatbot: Ask for Dishes by Cuisine & Price")
user_query = st.text_input("E.g.: Show me Italian dishes under 300")

def chatbot_response(user_input):
    user_input = user_input.lower()
    price_match = re.findall(r'\d+', user_input)
    cuisine_match = None
    for c in df['Cuisine'].unique():
        if c.lower() in user_input:
            cuisine_match = c
            break

    price_limit = int(price_match[0]) if price_match else None

    if cuisine_match and price_limit:
        result = df[(df['Cuisine']==cuisine_match) & (df['Price']<=price_limit)]
    elif cuisine_match:
        result = df[df['Cuisine']==cuisine_match]
    else:
        return "Please specify a cuisine or price range."

    if result.empty:
        return "No dishes found for your query."
    else:
        return result[['Dish_Name','Cuisine','Price','Rating']].head(5)

if user_query:
    response = chatbot_response(user_query)
    st.write(response)

# Model Evaluation
st.subheader("Model Evaluation")
st.write("Unique cuisines recommended:", df['Cuisine'].nunique())
st.write("Average Rating:", round(df['Rating'].mean(), 2))
