from flask import Flask, render_template, request, jsonify
from llm import qa_chain  

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.form.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        response = qa_chain.invoke({"query": user_query})
        answer = response["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in response["source_documents"]]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
