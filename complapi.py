from flask import Flask, request, jsonify, send_file
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from flask_caching import Cache
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

app = Flask(__name__)

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
template = """Use the following pieces of context to answer the question at the end.
Use as many of the provided words as possible to make a sentence. If the level is A1, the sentence is 5 words long. If the level is A2, the sentence is 6 words long.
If the level is B1, the sentence is 7 words long. Make sure the sentence is child-safe and appropriate.
Don't say anything that isn't a direct part of your answer. Take out one word from the sentence. The word must be in the provided list.
Replace it with ______. Then, separate the next part from the sentence
with a newline (\n). Take the word you replaced with ______ and add two other words separated by comma. The two other
words have to be in a similar semantic/syntactic category as the replaced word but must show some small differences.
Provide the sentence, then a newline (\n) and then the three words as described. Do not provide anything else.
{context}

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
        sentence = resultChain.split("\n")[0]
        optionList = resultChain.split("\n")[0]
        optionList = optionList.split(",")
        sentenceList = sentence.split(" ")
        if(level == "A1" and not len(sentenceList) == 5 and not "_"in sentence and not len(optionList)==3):
            while(len(sentenceList) != 5 and "_" not in sentence and not len(optionList) == 3):
                resultChain = rag_chain.invoke(stringConcat)
                sentence = resultChain.split("\n")[0]
                sentenceList = sentence.split(" ")
        if(level == "A2" and not len(sentenceList) == 6 and not "_"in sentence and not len(optionList)==3):
            while(len(sentenceList) != 6 and "_" not in sentence and not len(optionList) == 3):
                resultChain = rag_chain.invoke(stringConcat)
                sentence = resultChain.split("\n")[0]
                sentenceList = sentence.split(" ")
        if(level == "B1" and not len(sentenceList) == 7 and not "_"in sentence and not len(optionList)==3):
            while(len(sentenceList) != 7 and "_" not in sentence and not len(optionList) == 3):
                resultChain = rag_chain.invoke(stringConcat)
                sentence = resultChain.split("\n")[0]
                sentenceList = sentence.split(" ")
        optionList = resultChain.split("\n")[1]
        # Return response
        if(level=="A1"):
            return jsonify({
                'sentence1': sentenceList[0],
                'sentence2': sentenceList[1],
                'sentence3': sentenceList[2],
                'sentence4': sentenceList[3],
                'sentence5': sentenceList[4],
                'option1': optionList.split(",")[0],
                'option2': optionList.split(",")[1],
                'option3': optionList.split(",")[2],
            })
        if(level=="A2"):
            return jsonify({
                'sentence1': sentenceList[0],
                'sentence2': sentenceList[1],
                'sentence3': sentenceList[2],
                'sentence4': sentenceList[3],
                'sentence5': sentenceList[4],
                'sentence6': sentenceList[5],
                'option1': optionList.split(",")[0],
                'option2': optionList.split(",")[1],
                'option3': optionList.split(",")[2],
            })
        if(level=="B1"):
            return jsonify({
                'sentence1': sentenceList[0],
                'sentence2': sentenceList[1],
                'sentence3': sentenceList[2],
                'sentence4': sentenceList[3],
                'sentence5': sentenceList[4],
                'sentence6': sentenceList[5],
                'sentence7': sentenceList[6],
                'option1': optionList.split(",")[0],
                'option2': optionList.split(",")[1],
                'option3': optionList.split(",")[2],
            })

if __name__ == '__main__':
    app.run(debug=True)
