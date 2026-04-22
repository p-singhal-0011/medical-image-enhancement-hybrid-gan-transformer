import os
import torch
from flask import Flask, render_template, request
from model.hybrid_model import GeneratorUNetTransformer
from utils.preprocess import preprocess_image
from utils.postprocess import save_output

app = Flask(__name__)

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GeneratorUNetTransformer().to(device)
model.load_state_dict(torch.load("model/hybrid_best.pth", map_location=device))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        output_path = os.path.join(OUTPUT_DIR, file.filename)

        file.save(input_path)

        inp = preprocess_image(input_path).to(device)

        with torch.no_grad():
            out = model(inp)

        save_output(out, output_path)

        return render_template("index.html",
                               input_image=input_path,
                               output_image=output_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
