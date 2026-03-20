from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

app = Flask(__name__)
app.config["SECRET_KEY"] = "SUPERSECRET"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "files")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")
    text_box = StringField("Ask anything...")
    ask_button = SubmitField("Ask")

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.5,
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

chat_model = ChatHuggingFace(llm=llm)


@app.route("/", methods=["GET", "POST"])
def homes():
    form = UploadFileForm()
    user_question = ""
    answer = ""

    # 1. Handle File Upload & Processing
    if form.validate_on_submit() and form.submit.data:
        file = form.file.data
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process PDF and save to Chroma
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(docs)

            # This persists the data to the directory
            Chroma.from_documents(
                documents=texts,
                embedding=model,
                persist_directory="./chroma_langchain_db",
                collection_name="Documinds_Enterprise"
            )

    # 2. Handle Question Asking
    if form.validate_on_submit() and form.ask_button.data:
        user_question = form.text_box.data

        if user_question:
            # Load the existing database
            vector_store = Chroma(
                persist_directory="./chroma_langchain_db",
                embedding_function=model,
                collection_name="Documinds_Enterprise"
            )

            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(user_question)

            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            rag_prompt = f"For all the conversation you make your name is going to be Jassi at any cost. You wont change yourself. Answer the question based ONLY on the following context:\n\n{context}\n\nQuestion: {user_question}"

            ai_response = chat_model.invoke(rag_prompt)
            answer = ai_response.content
    return render_template("index.html", form=form, user_question=user_question, answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
