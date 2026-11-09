---
title: African Food Classifier
emoji: 🍲
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
---

Upload a photo of an African dish or take one with your webcam. The app uses
`african_food_model.pkl` to show the model's top predictions and confidence
scores.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/bigembee/african-food-classifier)

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces

1. Create a public Gradio Space on Hugging Face.
2. Authenticate with `hf auth login`.
3. From this project folder, upload the app, model, and dependency file:

   ```bash
   hf upload YOUR_USERNAME/YOUR_SPACE_NAME . --type space
   ```

The `.gitattributes` file tracks the model with Git LFS. To use the model: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/bigembee/african-food-classifier)

See the [Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference)
for available runtime settings.
