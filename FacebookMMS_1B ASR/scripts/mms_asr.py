import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "facebook/mms-1b-all"

LANG_MAP = {
    "as": "asm",           # Assamese (short code)
    "assamese": "asm",     # Assamese
    "bn": "ben",           # Bengali (short code)
    "bengali": "ben",      # Bengali
    "hin": "hin",          # Hindi (short code)
    "hindi": "hin",        # Hindi
    "ory": "ory",          # Oriya / Odia (short code)
    "odia": "ory",         # Odia
    "tam": "tam",          # Tamil (short code)
    "tamil": "tam",        # Tamil
    "tel": "tel",          # Telugu (short code)
    "telegu": "tel"        # Telugu (task-specified spelling)
}


class MMSASR:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()

    def transcribe(self, audio, sr, lang):
        if lang not in LANG_MAP:
            raise ValueError(f"Unsupported language: {lang}")

        mms_lang = LANG_MAP[lang]

        self.processor.tokenizer.set_target_lang(mms_lang)
        self.model.load_adapter(mms_lang)

        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        ids = torch.argmax(logits, dim=-1)
        return self.processor.decode(ids[0])
