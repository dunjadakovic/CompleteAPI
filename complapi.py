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

llm = ChatOpenAI(model="gpt-4o-mini")

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
template = """Generate a fill-in-the-gaps exercise with the following format based on the following topic {context}:
Provide a meaningful and grammatically correct sentence with a gap (eg "The cat ___ on the mat")
Below the sentence provide four options for the missing word. Ensure one option is correct and the others are not from the current topic but on a similar language level.
Rules for the exercise:
1. The sentence must be coherent and contextually meaningful.
2. The options should be related to the sentence, ensuring the distractors (wrong options) are fairly obviously incorrect, clearly distinguishable from the wrong answer and not from the current topic.
3. The correct option must make sense in the context of the sentence.
Output format: YOU MUST ADHERE TO THIS AT ALL COST OR NOTHING WILL WORK
Sentence \n option1, option2, option3, option4, option5
For example:
The cat ___ on the mat. \n sat, runs, sings, smiles, laughs


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
