from flask import Flask, request, jsonify, send_file
import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import requests
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set API key for OpenAI usage

app.config["SECRET_KEY"] = os.urandom(24)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY") 
modal_key = os.getenv("MODAL_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load CSV file
file_url = "https://raw.githubusercontent.com/dunjadakovic/RAGAPI/main/ContentAndCategories.csv"
response = requests.get(file_url)
with open("temp.csv", "wb") as f:
    f.write(response.content)

loader = CSVLoader(file_path="temp.csv")
data = loader.load()
# Create text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,  # length of longest category
    chunk_overlap=0,  # no overlap as complete separation, structured data
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(data)

# Create vector store
vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Define prompt template
template = """Use this topic {context} to create a fill in the gaps exercise with at least 5 options. It should be formatted like sentencestart_____sentenceend \n option1, option2, option3. Make sure to always provide a sentence that makes sense and at least 5 options in the given format.
ADHERE TO THE FORMAT AND GIVE LOTS OF OPTIONS!!!!

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | custom_rag_prompt
      | llm
      | StrOutputParser()
    )



@app.route('/api/generate', methods=['GET'])
def generate():
    level = request.args.get('level')
    topic = request.args.get('topic')
    if not level or not topic:
        return jsonify({'error': 'Missing level or topic'}), 400
    else:
        stringConcat = level + "," + topic
        resultChain = rag_chain.invoke(stringConcat)
        logging.info(f"Level: {level} Topic {topic}")
        options = None
        while options is None:
            try:
                sentence = resultChain.split("\n")[0]
                options = resultChain.split("\n")[1]
                optionList = options.split(",")
            except:
                resultChain = rag_chain.invoke(stringConcat)
        # Return response
        response = {
        "sentence": sentence
        }
        response.update({
            f"option{i+1}": optionList[i] for i in range(len(optionList))
        })
        response.update({
            "optionList": optionList
        })
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
