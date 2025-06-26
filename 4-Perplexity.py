import math

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# =============================================================================
#  Amaç:
# GPT-2 dil modeli kullanılarak iki farklı metnin (Türkçe ve İngilizce)
# - Cross-entropy (log loss)
# - Perplexity (şaşırma düzeyi)
# değerleri hesaplanmaktadır.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Giriş metinleri (Türkçe ve İngilizce)
# -----------------------------------------------------------------------------
turkish_text = "Bugün hava çok güzel ama akşam yağmur bekleniyor."
english_text = (
    "The weather is very nice today, but rain is expected in the evening."
)

# -----------------------------------------------------------------------------
# 2. GPT-2 tokenizer ve model yükleniyor
# -----------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # modeli değerlendirme moduna al


# -----------------------------------------------------------------------------
# 3. Cross-Entropy ve Perplexity hesaplayan fonksiyon
# -----------------------------------------------------------------------------
def compute_perplexity(text):
    """
    Verilen metnin GPT-2 modeli altında cross-entropy ve perplexity'sini hesaplar.

    Parametre:
    ----------
    text : str
        Değerlendirilecek doğal dil cümlesi

    Dönüş:
    ------
    loss : float
        Ortalama kelime başına negatif log-likelihood (cross-entropy)
    perplexity : float
        Modelin kelime tahminlerindeki belirsizlik (e^loss)
    """
    # Metni token haline getir (PyTorch tensörü olarak)
    inputs = tokenizer(text, return_tensors="pt")

    # Model çıktısını hesapla, hedef olarak girişin kendisini veriyoruz (language modeling)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())

    return loss.item(), perplexity


# -----------------------------------------------------------------------------
# 4. Türkçe ve İngilizce metinler için sonuçları hesapla
# -----------------------------------------------------------------------------
tr_loss, tr_perplexity = compute_perplexity(turkish_text)
en_loss, en_perplexity = compute_perplexity(english_text)

# -----------------------------------------------------------------------------
# 5. Sonuçları yazdır
# -----------------------------------------------------------------------------
print("Türkçe Metin:")
print(f" Cross-Entropy (Loss): {tr_loss:.4f}")
print(f" Perplexity:           {tr_perplexity:.4f}\n")

print("İngilizce Metin:")
print(f" Cross-Entropy (Loss): {en_loss:.4f}")
print(f" Perplexity:           {en_perplexity:.4f}")
