import os
import sqlite3
import pandas as pd
from groq import Groq
import re
from pathlib import Path
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL_NAME")

db_path = Path(__file__).parent / "db.sqlite"

client_sql = Groq()

sql_prompt = f"""
        You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
        pertaining to the data you have. The schema is provided in the schema tags. 
        <schema> 
        table: product 

        fields: 
        product_link - string (hyperlink to product)	
        title - string (name of the product)	
        brand - string (brand of the product)	
        price - integer (price of the product in Indian Rupees)	
        discount - float (discount on the product. 10 percent discount is represented as 0.1, 20 percent as 0.2, and such.)	
        avg_rating - float (average rating of the product. Range 0-5, 5 is the highest.)	
        total_ratings - integer (total number of ratings for the product)

        </schema>
        Make sure whenever you try to search for the brand name, the name can be in any case. 
        So, make sure to use %LIKE% to find the brand in condition. Never use "ILIKE". 
        Create a single SQL query for the question provided. 
        The query should have all the fields in SELECT clause (i.e. SELECT *)

        Just the SQL query is needed, nothing more. Always provide the SQL in between the <SQL></SQL> tags.
    """

comprehension_prompt = """
You are an expert in understanding the context of the question and replying based on the data pertaining to the question provided. You will be provided with Question: and Data:. The data will be in the form of an array or a dataframe or dict. Reply based on only the data provided as Data for answering the question asked as Question. Do not write anything like 'Based on the data' or any other technical words. Just a plain simple natural language response.
The Data would always be in context to the question asked. For example is the question is “What is the average rating?” and data is “4.3”, then answer should be “The average rating for the product is 4.3”. So make sure the response is curated with the question and data. Make sure to note the column names to have some context, if needed, for your response.
There can also be cases where you are given an entire dataframe in the Data: field. Always remember that the data field contains the answer of the question asked. All you need to do is to always reply in the following format when asked about a product: 
Produt title, price in indian rupees, discount, and rating, and then product link. Take care that all the products are listed in list format, one line after the other. Not as a paragraph.
For example:
1. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
2. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
3. Campus Women Running Shoes: Rs. 1104 (35 percent off), Rating: 4.4 <link>
Always reply in a very concise manner. Never add any extra information.
"""


def generate_sql_query(question):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sql_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
    )
    answer = chat_completion.choices[0].message.content
    return answer


def run_query(query):
    if query.strip().upper().startswith("SELECT"):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df


def sql_chain(question: str):
    sql_query = generate_sql_query(question)
    sql_match = re.search(r"<SQL>(.*?)</SQL>", sql_query, re.DOTALL)
    if sql_match is None:
        raise ValueError(
            "No SQL query found in the response. LLM is not able to generate the SQL query."
        )
    sql_text = sql_match.group(1).strip()
    # Basic validation: ensure it's a SELECT query
    if not sql_text.upper().startswith("SELECT"):
        raise ValueError(f"Generated SQL does not start with SELECT: {sql_text}")
    print("Generated SQL Query:", sql_text)
    response = run_query(sql_text)
    if response is None:
        raise ValueError("Sorry there was a problem in executing the SQL query.")

    context = response.to_dict(orient="records")
    answer = data_comprehention(question, context)
    return answer


def data_comprehention(question, context):
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": comprehension_prompt,
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\nDATA: {context}",
            },
        ],
        model=GROQ_MODEL,
        temperature=0.2,
    )
    answer = chat_completion.choices[0].message.content
    return answer


if __name__ == "__main__":
    question = "Give me PUMA shoes with rating above 4.5 and more than 30% discount."
    result = sql_chain(question)
    print("\n", result)
