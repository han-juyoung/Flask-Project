# from importlib_metadata import method_cache
from base64 import decode
from flask_wtf import FlaskForm
from pathlib import Path
import omp_crush
import numpy as np
import sys
import torch
from flask import Flask, request, jsonify, render_template, url_for
import json, requests
from model.model import MilabModel
import pathlib
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from data_loader.data_preprocessor import DataPreprocesser


omp_crush
app = Flask(__name__)

IMG_FOLDER = os.path.join("static", "image")
CSS_FOLDER = os.path.join("static", "css")

app.config["UPLOAD_FOLDER"] = IMG_FOLDER
app.config["UPLOAD_CSS_FOLDER"] = CSS_FOLDER


@app.route("/")
def index():
    return render_template("FormPage.html")


def on_json_loading_failed_return_dict(view):
    return {}


@app.route("/view", methods=["POST", "GET"])
def view():
    if request.method == "POST":
        view = request.form.to_dict(type=int)
        print(view)
        return render_template("ViewData.html", view=view)


def run_inference(in_tensor):
    model = MilabModel([23, 90, 80, 90])
    model.load_state_dict(
        torch.load(
            "saved/models/Milab_project_no_1/1006_004540/model_best_0.pth",
            map_location="cpu",
        )["state_dict"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    in_tensor = in_tensor.to(device)
    with torch.no_grad():
        out_tensor = model(in_tensor).squeeze(0)

    probs = F.softmax(out_tensor)[1] * 100
    print("probs : ", probs)
    out = probs.to("cpu").numpy()
    print("out", out)
    return out


def make_request():
    res = requests.get("https://reqres.in/api/users")
    parsed = res.json()
    print(type(parsed))
    for entry in parsed:
        print(entry)


def draw_histo(out):
    df = pd.read_excel("sig200.xlsx")
    thresh = 27.48091
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.set_palette("Set2")
    sns.kdeplot(
        data=df,
        x="Probability",
        ax=ax,
        hue="df_class",
        legend=False,
        multiple="stack",
        common_norm=False,
    )

    font = {"family": "Arial", "weight": "normal", "size": 14}
    font2 = {"family": "Arial", "weight": "bold", "size": 16}

    plt.xlim(0, 100)
    plt.xlabel("Probability", labelpad=10, fontdict=font2)
    plt.ylabel("Distribution", labelpad=10, fontdict=font2)

    # threshold_line
    ax.axvline(thresh, color="black", linewidth=3)
    str_thresh = "Threshold: " + str(thresh)
    plt.text(30, 0.05, str_thresh, fontdict=font)  # threshold text

    h_risk = "Predict : High-risk Pregnant"
    l_risk = "Predict : Low-risk Pregnant"

    if out < thresh:
        pat_text = "Subject:" + str(out)
        ax.axvline(out, color="royalblue", linewidth=3)
        ax.text(out + 2, 0.04, pat_text, color="blue", fontdict=font)
        ax.text(70, 60, l_risk, color="blue")

    if out > thresh:
        pat_text = "Subject:" + str(out)
        ax.axvline(out, color="red", linewidth=3)
        ax.text(out + 2, 0.04, pat_text, color="red", fontdict=font)
        ax.text(70, 60, h_risk, color="red")

    # 그림저장
    fig.savefig("src/static/image/HISTO.png")


class Quadrangle:
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color

    def get_area(self):
        return self.width * self.height


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        view = request.form.to_dict()
        print("view", view)
        print("type(view)", type(view))

        for i, j in view.items():
            view[i] = float(j)

        dp = DataPreprocesser(view)
        a = dp.scale()
        print("type(a):", type(a))
        in_tensor = torch.from_numpy(a)
        in_tensor = in_tensor.type(torch.float32)
        print("in_tensor.squeeze(0):", in_tensor.squeeze(0))
        out = run_inference(in_tensor)
        print("type: ", type(out))
        print("output", out)
        out_2 = np.round(out, 2)
        draw_histo(out)
        Plot_img = os.path.join(app.config["UPLOAD_FOLDER"], "HISTO.png")
        Pred_img = os.path.join(app.config["UPLOAD_FOLDER"], "white.PNG")
        circle = Quadrangle(10, 5, "red")

        return render_template(
            "PrintPage.html",
            plot_image=Plot_img,
            predict_image=Pred_img,
            out=out_2,
            circle_img=circle,
        )


if __name__ == "__main__":
    app.run(debug=True)
    print(sys.argv[1])
