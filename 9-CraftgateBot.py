import os

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# 1. Ortam Değişkeni ve Model Ayarları
# =============================================================================

# .env dosyasından API anahtarını al
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Kullanılacak modeller
GENERATOR_MODEL = "gpt-3.5-turbo"
FALLBACK_MODEL = "gpt-4"
JUDGE_MODEL = "gpt-4"


# =============================================================================
# 2. Yanıt Üretimi (Zayıf Model - GPT-3.5)
# =============================================================================
def generate_response(prompt: str) -> str:
    """
    GPT-3.5 modelini kullanarak müşteri sorusuna ilk yanıtı üretir.
    """
    completion = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen Craftgate müşteri destek asistanısın. "
                    "Craftgate POS hizmeti sunmaktadır. Buna göre cevap ver."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 3. Kendi Kendini Değerlendirme (Self-Evaluation)
# =============================================================================
def self_evaluate(prompt: str, response: str) -> str:
    """
    Yanıtın doğru olup olmadığını GPT-4 ile kontrol eder.
    Doğru değilse düzeltme gerekçesiyle birlikte "Yanlış" döner.
    """
    eval_prompt = f"""
Aşağıda bir müşteri sorusu ve yapay zekanın verdiği yanıt bulunmaktadır.

Craftgate, ödeme orkestrasyon firmasıdır ve POS cihazı sağlamamaktadır.

Verilen yanıt doğru mu kontrol et. Yanlışsa belirt.

Müşteri sorusu: {prompt}
Yapay zeka yanıtı: {response}

Lütfen şu formatta cevapla:
Değerlendirme: Doğru ya da Yanlış
Açıklama: [değerlendirme gerekçesi]
""".strip()

    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Sen kendini değerlendiren bir yapay zekasın."
            },
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 4. Fallback (Daha Güçlü Modelle Yeni Yanıt - GPT-4)
# =============================================================================
def fallback_response(prompt: str) -> str:
    """
    Yanıt yanlışsa, GPT-4 kullanılarak doğru ve açık bir cevap üretilir.
    """
    completion = client.chat.completions.create(
        model=FALLBACK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen Craftgate müşteri destek uzmanısın. "
                    "Açık, doğru ve kısa cevaplar ver."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 5. Yanıt Kalitesini Puanlama (Reward Scoring)
# =============================================================================
def reward_score(prompt: str, response: str) -> str:
    """
    Verilen yanıtı GPT-4 ile aşağıdaki kriterlere göre puanlar:
    - Doğruluk
    - Açıklık
    - Craftgate'e uygunluk
    """
    judge_prompt = f"""
Aşağıda bir müşteri sorusu ve buna verilen bir destek yanıtı bulunmaktadır.

Bu yanıtı aşağıdaki kriterlere göre değerlendir:

- Yanıt doğruluğu
- Açıklık
- Craftgate'e uygunluk

Her kriteri değerlendir ve genel bir açıklama yap.
Ardından 1 ile 5 arasında genel bir puan ver.

Müşteri sorusu: {prompt}
Yanıt: {response}

Cevap formatı:
Açıklama: [detaylı açıklama yaz]
Puan: [1-5 arası sayı]
""".strip()

    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Sen müşteri destek kalitesi değerlendiren bir yapay zekasın."
            },
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content.strip()


# =============================================================================
# 6. Ana Uygulama Akışı
# =============================================================================
if __name__ == "__main__":
    user_prompt = "Craftgate olarak POS veriyor musunuz"

    print("Müşteri sorusu alındı")
    print(user_prompt)

    print("\nYanıt üretiliyor")
    first_response = generate_response(user_prompt)
    print("Zayıf modelin yanıtı")
    print(first_response)

    print("\nYanıt kendi içinde değerlendiriliyor")
    self_judgment = self_evaluate(user_prompt, first_response)
    print("Kendi kendini değerlendirme sonucu")
    print(self_judgment)

    if "Yanlış" in self_judgment:
        print("\nYanıt yanlış bulundu, fallback başlatılıyor")
        better_response = fallback_response(user_prompt)
        print("Güçlü modelden düzeltme yanıtı")
        print(better_response)

        print("\nDüzeltme yanıtı puanlanıyor")
        score = reward_score(user_prompt, better_response)
        print("Değerlendirme sonucu")
        print(score)
    else:
        print("\nYanıt doğru, düzeltme gerekmedi")
