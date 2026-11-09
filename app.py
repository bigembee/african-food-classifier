import os
import pathlib
import sys
import types
from pathlib import Path

import spaces
import gradio as gr
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from PIL import Image
import torch


pathlib_local = types.ModuleType("pathlib._local")
pathlib_local.Path = pathlib.Path
pathlib_local.PosixPath = (
    pathlib.PosixPath if os.name != "nt" else pathlib.WindowsPath
)
sys.modules.setdefault("pathlib._local", pathlib_local)

MODEL_PATH = Path(__file__).with_name("african_food_model.pkl")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

learner = load_learner(MODEL_PATH, cpu=True)
learner.model.eval()
learner.dls.valid.after_item.split_idx = 1
learner.dls.valid.after_batch.split_idx = 1


@spaces.GPU
def classify_food(image: Image.Image | None) -> dict[str, float]:
    if image is None:
        raise gr.Error("Upload or capture an image of food to classify.")

    sample = PILImage.create(image)
    sample = learner.dls.valid.after_item(sample)
    sample = learner.dls.valid.before_batch(sample)
    batch = learner.dls.valid.after_batch(sample.unsqueeze(0))

    with torch.inference_mode():
        logits = learner.model(batch)
        probabilities = torch.softmax(logits, dim=-1)[0]

    return {
        str(label): float(probability)
        for label, probability in zip(learner.dls.vocab, probabilities)
    }


with gr.Blocks(title="African Food Classifier") as demo:
    gr.Markdown(
        """
        # African Food Classifier
        Upload a food photo or take one with your camera to see the model's
        top predictions and confidence scores.
        """
    )

    with gr.Row():
        food_image = gr.Image(
            label="Food photo",
            type="pil",
            sources=["upload", "webcam"],
        )
        predictions = gr.Label(label="Predictions", num_top_classes=5)

    classify_button = gr.Button("Classify food", variant="primary")
    classify_button.click(
        fn=classify_food,
        inputs=food_image,
        outputs=predictions,
    )

    gr.Markdown(
        "Predictions are model estimates and may be uncertain for unfamiliar "
        "dishes, mixed foods, or low-quality images."
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())