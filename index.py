from flask import Flask, render_template, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import PyPDF2
import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK tokenizer (only once)
nltk.download('punkt')

app = Flask(__name__)

# === 1. ChatterBot Setup ===
bot = ChatBot(
    "AntiDopingBot",
    read_only=True,
    logic_adapters=[{
        "import_path": "chatterbot.logic.BestMatch",
        "default_response": "Sorry, I cannot get you. Please visit https://adak.go.ke or call +254-20-2724208.",
        "maximum_similarity_threshold": 0.9
    }]
)

# Train ChatterBot
list_trainer = ListTrainer(bot)
list_trainer.train([
    "hi", "Hi there! How can I help you?",
    "what is doping", "Doping refers to the use of prohibited substances or methods to enhance performance.",
    "report doping", "Visit https://adak.go.ke or call ADAK anonymously at +254-20-2724208.",
    "banned substances", "Find the list on WADA's website: https://www.wada-ama.org/",
    "adak contact", "Email: info@adak.go.ke or phone: +254-20-2724208.",
    "testing process", "Testing includes sample collection, lab analysis, and result management.",
    "athlete rights", "Athletes have the right to legal counsel, fair hearing, and privacy."
])

corpus_trainer = ChatterBotCorpusTrainer(bot)
corpus_trainer.train("chatterbot.corpus.english")

# === 2. Text Chunking (manual, no LangChain needed) ===
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === 3. Extract and Chunk PDF ===
def extract_pdf_chunks(path):
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            raw = page.extract_text()
            if raw:
                for line in raw.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("page") or "http" in line or len(line) < 5:
                        continue
                    text += line + "\n"
    return chunk_text(text)

# === 4. Embed PDF Chunks ===
pdf_path = os.path.join("pdfs", "adak.pdf")  # Make sure this file exists
pdf_chunks = extract_pdf_chunks(pdf_path)
model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_embeddings = model.encode(pdf_chunks)

# === 5. Semantic Search Function ===
def search_pdf(query, top_n=3):
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, pdf_embeddings)[0]
    best_indices = similarity.argsort()[::-1][:top_n]

    if similarity[best_indices[0]] < 0.4:
        return "No relevant information found in the ADAK document."

    return "\n\n".join([pdf_chunks[i] for i in best_indices])

# === 6. Flask Routes ===
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    user_input = request.json["msg"]
    bot_response = str(bot.get_response(user_input))
    doc_response = search_pdf(user_input)
    return jsonify({
        "response": f"ADAK SUPPORT: {bot_response}\n\n📄 From ADAK PDF:\n{doc_response}"
    })

if __name__ == "__main__":
    app.run(debug=True)