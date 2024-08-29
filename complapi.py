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
from supabase import create_client, Client
from io import BytesIO
from modal import Stub 

app = Flask(__name__)

# Set API key for OpenAI usage

app.config["SECRET_KEY"] = os.urandom(24)
load_dotenv()
secrets = [modal.Secret.from_name("complapisecrets")]

openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY") 
modal_key = os.getenv("MODAL_API_KEY")
bucket_name = 'ttsapi'

stub = Stub(modal_key)

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
Use as many of the provided words as possible to make a sentence. The sentence has to be exactly 5 words long. Make sure the sentence is child-safe and appropriate.
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

preload_models()

# Create Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

@app.route('/api/generate', methods=['GET'])
def generate():
    level = request.args.get('level')
    topic = request.args.get('topic')
    if not level or not topic:
        return jsonify({'error': 'Missing level or topic'}), 400
    else:
        stringConcat = level + "," + topic
        resultChain = rag_chain.invoke(stringConcat)
        while "_" not in resultChain:
            resultChain = rag_chain.invoke(stringConcat)
        logging.info(f"Level: {level} Topic {topic}")

        # Generate audio
        text_prompt = resultChain.split("\n")[0]
        history_prompt = "v2/en_speaker_5"
        audio_array = stub.generate_audio(text_prompt, history_prompt, async = True)

        # Save audio to file
        audio_file = 'audio.wav'
        buffer = BytesIO()
        write(buffer, SAMPLE_RATE, audio_array)
        with open(audio_file, 'wb') as f:
            f.write(buffer.getvalue())

        # Upload audio file to Supabase
        supabase.storage.from_(bucket_name).upload(audio_file, buffer.getvalue(), 'audio/wav')
        audio_path = supabase.storage.from_(bucket_name).create_signed_url(audio_file, 60)
        audio_path = supabase.storage.from_(bucket_name).create_signed_url(audio_file, 60 * 60)

        # Return response
        return jsonify({
            'sentence': resultChain.split("\n")[0],
            'options': resultChain.split("\n")[1].split(","),
            'audio': audio_path
        })

if __name__ == '__main__':
    app.run(debug=True)
