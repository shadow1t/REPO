# ML Starter Template

A machine learning project starter with notebooks, structured source code, and core ML/data packages.

## 📦 Structure

- `notebooks/` — Jupyter notebooks
- `src/` — Source code modules
- `data/` — Raw datasets
- `models/` — Trained model artifacts
- `tests/` — Unit tests

## ▶️ Usage

1. Activate environment:
   ```bash
   source venv/bin/activate
   ```

2. Start notebook:
   ```bash
   jupyter notebook
   ```

3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

---

# NASA Chat AI

A simple, free, open-source chat assistant that uses NASA public data and explains it in simple language. Supports Arabic and English, can optionally caption images, and simplifies technical text for non-experts (kids included).

## Features

- Uses NASA public APIs (APOD, EPIC, Image and Video Library)
- Text questions: searches NASA and returns context + simplified explanation
- Image questions: captions locally with BLIP and finds related NASA context
- Simplifies technical text using FLAN‑T5
- Translates answers to Arabic or English

## Quickstart

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Set your NASA API key (optional but recommended)
   - Create a `.env` file and add:
     ```
     NASA_API_KEY=YOUR_KEY_HERE
     ```
   - Or export in your shell:
     ```bash
     export NASA_API_KEY=YOUR_KEY_HERE
     ```

3. Run the chat app (Streamlit)
   ```bash
   streamlit run app.py
   ```

4. Use the sidebar to choose:
   - Answer language: Arabic (`ar`) or English (`en`)
   - Image understanding: enable/disable BLIP captioner

## Code Overview

- `src/nasa_chat/nasa_api.py` — minimal NASA API client
- `src/nasa_chat/translator.py` — English ↔ Arabic translator (MarianMT)
- `src/nasa_chat/simplifier.py` — simplifier using FLAN‑T5
- `src/nasa_chat/captioner.py` — image captioning with BLIP
- `src/nasa_chat/chatbot.py` — orchestration for the chat logic
- `app.py` — Streamlit UI

## Notes

- Running models on CPU is possible but may be slow for large inputs.
- If you only need Arabic, you can keep the translator configured for `en→ar`.
- All models used are free and available on Hugging Face.

## Arabic Summary (ملخص بالعربية)

- هذا مشروع بوت ذكاء اصطناعي يعتمد على بيانات ناسا المفتوحة.
- يبسّط المعلومات المعقّدة لتصبح مفهومة للأطفال والناس العاديين.
- يدعم اللغة العربية والإنجليزية، ويمكنه وصف الصور (اختياري).
- للتشغيل: ثبّت الحزم، أضف مفتاح ناسا إن وجد، ثم نفّذ: `streamlit run app.py`.

