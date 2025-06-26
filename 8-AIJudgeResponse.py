import os

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# 1. Ortam Değişkeni ve API Başlatma
# =============================================================================

# .env dosyasındaki API anahtarını yükle
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Üretim ve değerlendirme için kullanılacak modeller
GENERATOR_MODEL = "gpt-3.5-turbo"
JUDGE_MODEL = "gpt-4"


# =============================================================================
# 2. Yanıt Üretimi Fonksiyonu
# =============================================================================
def generate_response(prompt: str) -> str:
    """
    GPT-3.5-turbo ile verilen kullanıcı girdisine karşılık
    yardımcı bir yapay zeka cevabı üretir.
    """
    completion = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Sen yardımcı bir yapay zekasın."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 3. Tek Yanıt Üzerinden Değerlendirme (Doğruluk, uygunluk, netlik)
# =============================================================================
def judge_response(prompt: str, response: str) -> str:
    """
    GPT-4 ile verilen bir yanıtı; doğruluk, uygunluk ve anlaşılabilirlik
    kriterlerine göre değerlendirir ve puanlandırır (1–5 arası).
    """
    evaluation_prompt = f"""
Aşağıda bir kullanıcı girdisi ve bu girdiye verilen bir yanıt bulunmaktadır.

Lütfen bu yanıtı aşağıdaki kriterlere göre değerlendir:
- Yanıt doğruluğu
- Uygunluk
- Anlaşılabilirlik

Her bir kriteri değerlendir ve genel bir açıklama ver.
Son olarak genel bir puan ver. Puan 1 ile 5 arasında olmalıdır.

Kullanıcı girişi: {prompt}
Verilen yanıt: {response}

Lütfen şu formatta cevapla:
Açıklama: [buraya detaylı açıklama yaz]
Puan: [1-5 arası sayı]
""".strip()

    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Sen bir değerlendirme uzmanı yapay zekasın."
            },
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 4. Yanıt Karşılaştırma (A mı B mi daha iyi?)
# =============================================================================
def preference_judge(prompt: str, response_a: str, response_b: str) -> str:
    """
    GPT-4 modeline iki farklı cevabı vererek hangisinin daha iyi olduğunu
    belirlemesini sağlar. Cevap formatı: "Tercih edilen yanıt: A ya da B".
    """
    compare_prompt = f"""
Aşağıda bir kullanıcı girdisi ve bu girdiye verilmiş iki farklı yanıt bulunmaktadır.

Lütfen hangi yanıtın daha iyi olduğunu belirt.
Açıkla ve yalnızca A ya da B olarak sonucu belirt.

Kullanıcı girişi: {prompt}

Yanıt A: {response_a}

Yanıt B: {response_b}

Lütfen şu formatta cevapla:
Tercih edilen yanıt: A ya da B
Açıklama: [neden bu yanıtı seçtiğini yaz]
""".strip()

    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Sen bir karşılaştırmalı değerlendirme yapay zekasısın."
            },
            {"role": "user", "content": compare_prompt}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 5. Uygulama Girişi – Ana Akış
# =============================================================================
if __name__ == "__main__":
    user_prompt = "Türkiye'nin başkenti neresidir"

    print("Yanıt üretimi başlatılıyor...")
    generated_answer = generate_response(user_prompt)
    print("\nYanıt A üretildi:")
    print(generated_answer)

    answer_a = generated_answer
    answer_b = "Ankara Türkiye'nin başkentidir"
    print("\nYanıt B hazırlandı:")
    print(answer_b)

    print("\nYanıt değerlendirmesi başlatılıyor...")
    judgment = judge_response(user_prompt, generated_answer)
    print("\nYanıt A için değerlendirme sonucu:")
    print(judgment)

    print("\nKarşılaştırmalı değerlendirme başlatılıyor...")
    comparison = preference_judge(user_prompt, answer_a, answer_b)
    print("\nKarşılaştırma sonucu:")
    print(comparison)
