import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Neo4j graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Initialize ChatGroq model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

# Create the QA chain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

# Streamlit app layout
st.title("Graph Database Question Answering")

# User input for the query
query = st.text_input("Enter your question about the movie database:")

if query:
    # Get the response from the QA chain
    response = chain.invoke({"query": query})
    
    # Display the response
    st.write("Response:")
    st.write(response['result'])

    # Optionally, show the generated Cypher query
    st.write("Generated Cypher Query:")
    st.write(response.get('generated_cypher', 'No Cypher query generated.'))
