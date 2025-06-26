import os

from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasından API anahtarını yükle
load_dotenv()

# OpenAI istemcisi başlatılıyor
MODEL_NAME = "gpt-4"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# =============================================================================
# 1. Tek cevaplı değerlendirme (puanlama)
# =============================================================================
def generate_judge_prompt(question: str, answer: str) -> str:
    """
    Soruyu ve cevabı alır, değerlendirme prompt'unu oluşturur.
    """
    prompt = f"""
Aşağıda bir soru ve bu soruya verilen bir yanıt bulunmaktadır.

Senin görevin, verilen yanıtın soruya ne kadar uygun olduğunu değerlendirmektir.

Lütfen 1 ile 5 arasında bir puan ver. Açıklamanı da yaz.

- 1: Hiç uygun değil
- 2: Zayıf bir uygunluk
- 3: Kısmen uygun
- 4: Uygun ama küçük eksikleri var
- 5: Tam anlamıyla uygun

Soru: {question}

Yanıt: {answer}

Cevap formatı şu şekilde olsun:
PUAN: [buraya sayı yaz]
AÇIKLAMA: [buraya açıklama yaz]
"""
    return prompt.strip()


def evaluate_answer(question: str, answer: str) -> dict:
    """
    Verilen cevap için GPT-4 kullanarak puanlama yapar.
    """
    prompt = generate_judge_prompt(question, answer)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Sen bir yapay zeka değerlendirme uzmanısın."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    output = response.choices[0].message.content
    result = {"raw_output": output}

    for line in output.split("\n"):
        if line.startswith("PUAN:"):
            result["puan"] = line.replace("PUAN:", "").strip()
        if line.startswith("AÇIKLAMA:"):
            result["aciklama"] = line.replace("AÇIKLAMA:", "").strip()

    return result


# =============================================================================
# 2. Referans cevaba göre benzerlik değerlendirmesi
# =============================================================================
def evaluate_with_reference(
        question: str,
        reference_answer: str,
        generated_answer: str
) -> dict:
    """
    Üretilen cevabı referans cevap ile karşılaştırarak
    benzerlik derecesi belirler.
    """
    prompt = f"""
Aşağıda bir soru, bir referans yanıt ve bu soruya verilen bir başka yanıt bulunmaktadır.

Görevin, verilen cevabın referans cevaba ne kadar benzediğini değerlendirmektir.

Sadece şu şekilde cevap ver:
BENZER Mİ: True ya da False yaz
AÇIKLAMA: [neden benzer veya değil, açıklayıcı yaz]

Soru: {question}

Referans yanıt: {reference_answer}

Verilen yanıt: {generated_answer}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Sen referans bazlı bir yapay zeka yargıcısın."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    output = response.choices[0].message.content
    result = {"raw_output": output}

    for line in output.split("\n"):
        if line.startswith("BENZER Mİ:"):
            result["benzer"] = (
                    line.replace("BENZER Mİ:", "").strip() == "True"
            )
        if line.startswith("AÇIKLAMA:"):
            result["aciklama"] = line.replace("AÇIKLAMA:", "").strip()

    return result


# =============================================================================
# 3. İki cevabı kıyaslayarak daha iyisini seçme
# =============================================================================
def evaluate_comparison(
        question: str,
        answer_a: str,
        answer_b: str
) -> dict:
    """
    Aynı soruya verilmiş iki cevabı kıyaslar ve daha iyi olanı seçer.
    """
    prompt = f"""
Aşağıda bir soru ve bu soruya verilmiş iki farklı cevap bulunmaktadır.

Senin görevin, hangi cevabın daha iyi olduğunu belirlemektir.

Sadece şu şekilde cevap ver:
KAZANAN: A ya da B yaz
AÇIKLAMA: [neden o cevabın daha iyi olduğunu açıkla]

Soru: {question}

A şıkkı: {answer_a}

B şıkkı: {answer_b}
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Sen cevap karşılaştırması yapan bir değerlendirme uzmanısın."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    output = response.choices[0].message.content
    result = {"raw_output": output}

    for line in output.split("\n"):
        if line.startswith("KAZANAN:"):
            result["kazanan"] = line.replace("KAZANAN:", "").strip()
        if line.startswith("AÇIKLAMA:"):
            result["aciklama"] = line.replace("AÇIKLAMA:", "").strip()

    return result


# =============================================================================
# 4. Test senaryoları (manuel)
# =============================================================================
if __name__ == "__main__":
    question = "Türkiye'nin başkenti neresidir?"
    answer_a = "İstanbul Türkiye'nin başkentidir."
    answer_b = "Ankara Türkiye'nin başkentidir."
    reference_answer = "Ankara Türkiye'nin başkentidir."

    print("1. Tek cevaplı değerlendirme:")
    result_single = evaluate_answer(question, answer_a)
    print("Verilen puan:", result_single.get("puan"))
    print("Açıklama:", result_single.get("aciklama"))
    print()

    print("2. Referans cevap ile karşılaştırmalı değerlendirme:")
    result_reference = evaluate_with_reference(
        question, reference_answer, answer_a
    )
    print("Referansla benzer mi:", result_reference.get("benzer"))
    print("Açıklama:", result_reference.get("aciklama"))
    print()

    print("3. İki cevap arasında karşılaştırma:")
    result_comparison = evaluate_comparison(question, answer_a, answer_b)
    print("Tercih edilen cevap:", result_comparison.get("kazanan"))
    print("Açıklama:", result_comparison.get("aciklama"))
