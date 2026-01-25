from flask import Flask, request, jsonify, render_template, session
import requests
import json
import uuid
import os


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "26b14056a018e085068330a3283ec2f92c22abe18631a789750f2610e5eebbb0")


HF_API = "https://sonuramashish22028704-npmeduai.hf.space/ingestion"

@app.route("/")
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template("index.html")

class Memory:
    def __init__(self, user_id):
        self.filename = f"memory_{user_id}.json"

    def save_context(self, user_input, ai_output):
        with open(self.filename, "a") as data:
            json.dump({"Human": user_input, "AI": ai_output}, data)
            data.write("\n")

    def load_memory_variables(self):
        string_history = ""
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
            with open(self.filename, "r") as data:
                for line in data:
                    try:
                        ldata = json.loads(line)
                        string_history += f"Human: {ldata['Human']}\nAI: {ldata['AI']}\n"
                    except json.JSONDecodeError:
                        continue
        return string_history
        

@app.route("/ask", methods=["POST"])
def ask():
    user_id = session.get('user_id', str(uuid.uuid4()))
    memory = Memory(user_id)
    files = {}
    data = {}

    # Required
    data["instruction"]="""
    Hey You are an Legal Assistant and you have to follow following instructions:-
    1.Always provide correct and proven data 
    2.If you think the data that you have can be wrong then see the user data that you are getting
    3.If you cannot decide or you want more then you can ask user for more inf or as per your case
    4.Do not provide probable type data"""
    data["query"] = request.form.get("query")
    data["DB_PATH"] = request.form.get("DB_PATH")

    # Optional
    if "file" in request.files:
        f = request.files["file"]
        files["file"] = (f.filename, f.stream, f.mimetype)


    
    history = memory.load_memory_variables()
    full_prompt = f"Context history:\n{history}\nHuman: {data}\nAI:"
    res = requests.post(
        HF_API,
        data=data,
        files=files if files else None,
        timeout=600
    )
    response = str(result)
    memory.save_context(data, response)

    return jsonify({"response": res.json().get("response")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
