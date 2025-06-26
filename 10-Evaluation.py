import os
import random
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# 1. Ortam değişkeni ve OpenAI istemcisi
# =============================================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# 2. Değerlendirilecek prompt listesi
# =============================================================================
prompt_list = [
    "Yapay zeka nedir",
    "Python programlama dilinin avantajları nelerdir",
    "Makine öğrenmesi ile derin öğrenme arasındaki fark nedir"
]


# =============================================================================
# 3. Her prompt için iki farklı cevap üret
# =============================================================================
def generate_two_responses(prompt_text):
    """
    GPT-3.5 modelinden iki farklı sıcaklıkta yanıt üretir.
    """
    responses = []
    for _ in range(2):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Kısa ve açık bir şekilde Türkçe cevap ver"
                },
                {"role": "user", "content": prompt_text}
            ],
            temperature=random.uniform(0.5, 0.9),
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        responses.append(content)
    return responses


# =============================================================================
# 4. Kullanıcıya iki cevabı gösterip tercih ettir
# =============================================================================
def ask_user_to_compare(prompt, response_a, response_b):
    """
    Kullanıcıdan hangi cevabın daha iyi olduğunu girmesini ister.
    """
    print("\nSoru:")
    print(prompt)
    print("\nBirinci cevap:")
    print(response_a)
    print("\nİkinci cevap:")
    print(response_b)

    while True:
        choice = input(
            "\nHangi cevabı daha iyi buluyorsun? "
            "Birinci için 1, ikinci için 2 yaz: "
        ).strip()
        if choice in ["1", "2"]:
            return int(choice)
        print("Lütfen sadece 1 ya da 2 giriniz.")


# =============================================================================
# 5. Değerlendirme simülasyonu
# =============================================================================
results = []

# 3 karşılaştırma yap
for _ in range(3):
    prompt = random.choice(prompt_list)
    response_1, response_2 = generate_two_responses(prompt)
    selected = ask_user_to_compare(prompt, response_1, response_2)
    results.append({
        "prompt": prompt,
        "response_1": response_1,
        "response_2": response_2,
        "selected": f"response_{selected}"
    })

# =============================================================================
# 6. Model adları (her cevap aynı modelden geliyor gibi görünse de örnek için)
# =============================================================================
model_names = {
    "response_1": "gpt-3.5-turbo",
    "response_2": "gpt-4"
}

# Her model çiftine göre karşılaştırma sonucu tutulur
match_results = defaultdict(lambda: {"wins": 0, "total": 0})

# =============================================================================
# 7. Karşılaştırma sonuçlarını işleyerek istatistik oluştur
# =============================================================================
for result in results:
    selected = result["selected"]
    other = "response_2" if selected == "response_1" else "response_1"

    winner = model_names[selected]
    loser = model_names[other]

    # Kazanan model için istatistik güncelle
    key = (winner, loser)
    match_results[key]["wins"] += 1
    match_results[key]["total"] += 1

    # Kaybeden modelin de karşılaştırma sayısı güncellenmeli
    reverse_key = (loser, winner)
    match_results[reverse_key]["total"] += 1

# =============================================================================
# 8. Sonuçları yazdır
# =============================================================================
print("\nModel karşılaştırma istatistikleri:")
for (winner, loser), data in match_results.items():
    win_count = data["wins"]
    total_count = data["total"]
    win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
    print(
        f"{winner}, {loser} modeline karşı "
        f"ortalama %{win_rate:.1f} oranla tercih edildi"
    )

print("\nDeğerlendirme sonuçları:")
for index, item in enumerate(results, 1):
    print(f"\nSıra {index}")
    print("Soru:", item["prompt"])
    print("Seçilen cevap:", item[item["selected"]])
