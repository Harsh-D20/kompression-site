from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os

from kmeans import quantize_image

app = Flask(__name__)
load_dotenv()
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for production environment")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def quantize_interaction_manager():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash(
                "Invalid file type. Please upload an image (png, jpg, jpeg, gif, bmp)."
            )
            return redirect(request.url)

        if file:
            file = request.files["file"]
            img_pil = Image.open(file.stream).convert("RGB")
            original_img_array = np.array(img_pil)
            try:
                n_colors = int(request.form.get("k_colors", 16))
                if n_colors <= 0:
                    flash("Number of colors (k) must be a positive integer.")
                    return redirect(request.url)
                if (
                    n_colors > 256
                    and n_colors
                    > original_img_array.shape[0] * original_img_array.shape[1]
                ):
                    flash(
                        "Too many colors requested. k cannot be greater than the total number of pixels."
                    )
                    return redirect(request.url)
            except ValueError:
                flash("Invalid value for number of colors. Please enter an integer.")
                return redirect(request.url)

            try:
                max_iterations = int(request.form.get("max_iterations", 100))
                if max_iterations <= 0:
                    flash("Maximum iterations (n) must be a positive integer.")
                    return redirect(request.url)
            except ValueError:
                flash("Invalid value for max iterations. Please enter an integer.")
                return redirect(request.url)

            quantized_img_array, cluster_centers = quantize_image(
                original_img_array, n_colors, max_iterations
            )
            quantized_img_pil = Image.fromarray(quantized_img_array)
            original_img_b64 = image_to_base64(img_pil)
            quantized_img_b64 = image_to_base64(quantized_img_pil)
            results = {
                "original_img_b64": original_img_b64,
                "quantized_img_b64": quantized_img_b64,
                "k_value": n_colors,
                "original_dimensions": f"{original_img_array.shape[1]}x{original_img_array.shape[0]}",
            }
            return render_template("results.html", results=results)
    return render_template("index.html")


@app.route("/algorithm.html", methods=["GET"])
def algorithm_page():
    return render_template("algorithm.html")


def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


if __name__ == "__main__":
    app.run(debug=True)
