from flask import Flask, render_template, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer
import PyPDF2
import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Download NLTK resources for sentence tokenization ===
nltk.download('punkt')
try:
    nltk.download('punkt_tab')  # Needed for NLTK 3.9+ in some cases
except:
    pass

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

# === 2. Train ChatterBot with short, friendly answers ===
list_trainer = ListTrainer(bot)
list_trainer.train([
    # Greetings & small talk
    "hi", "Hi there! How can I help you today?",
    "hello", "Hello! How’s your day going?",
    "how are you", "I'm doing great, thanks for asking. How can I help?",
    "how was your day", "It's been great, thank you! How about yours?",
    "good morning", "Good morning! Hope you have a wonderful day ahead.",
    "good afternoon", "Good afternoon! How may I assist you?",
    "good evening", "Good evening! How can I help?",
    "thank you", "You're welcome! Happy to assist.",
    "thanks", "Anytime! 😊",

    # ADAK-specific
    "what is doping", "Doping is the use of banned substances or methods to enhance performance.",
    "report doping", "Go to https://adak.go.ke or call +254-20-2724208 anonymously.",
    "banned substances", "See WADA list: https://www.wada-ama.org/",
    "adak contact", "Email info@adak.go.ke or phone +254-20-2724208.",
    "testing process", "Sample collection, lab analysis, and result management.",
    "athlete rights", "Right to counsel, fair hearing, and privacy."
])

corpus_trainer = ChatterBotCorpusTrainer(bot)
corpus_trainer.train("chatterbot.corpus.english")

# === 3. Text Chunking ===
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === 4. Extract and Chunk PDF ===
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

# === 5. Embed PDF Chunks ===
pdf_path = os.path.join("pdfs", "adak.pdf")  # Ensure the PDF exists here
pdf_chunks = extract_pdf_chunks(pdf_path)
model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_embeddings = model.encode(pdf_chunks)

# === 6. Search PDF and return concise sentences ===
def search_pdf(query, top_n=3):
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, pdf_embeddings)[0]
    best_indices = similarity.argsort()[::-1][:top_n]

    if similarity[best_indices[0]] < 0.4:
        return "No relevant information found in the ADAK document."

    # Combine best chunks
    relevant_text = " ".join([pdf_chunks[i] for i in best_indices])

    # Break into sentences
    sentences = sent_tokenize(relevant_text)

    # Filter sentences containing query keywords
    keywords = query.lower().split()
    filtered_sentences = [
        s.strip() for s in sentences if any(k in s.lower() for k in keywords)
    ]

    # If no keyword match, take top sentences
    if not filtered_sentences:
        filtered_sentences = [s.strip() for s in sentences[:4]]

    # Limit to 4 sentences max
    brief_sentences = filtered_sentences[:4]

    # Return as sentences with spacing
    return "\n\n".join(brief_sentences)

# === 7. Shorten Bot Response ===
def concise_response(text, max_sentences=2):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:max_sentences])

# === 8. Flask Routes ===
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    user_input = request.json["msg"]
    bot_response = concise_response(str(bot.get_response(user_input)))
    doc_response = search_pdf(user_input)
    return jsonify({
        "response": f"ADAK SUPPORT: {bot_response}\n\n📄 From ADAK PDF:\n{doc_response}"
    })

if __name__ == "__main__":
    app.run(debug=True)
