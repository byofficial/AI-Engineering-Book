import os
import random
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# 1. Ortam değişkeni yüklenir ve OpenAI istemcisi başlatılır
# =============================================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# 2. Değerlendirme yapılacak soru listesi
# =============================================================================
prompt_list = [
    "Yapay zeka nedir",
    "Python programlama dilinin avantajları nelerdir",
    "Makine öğrenmesi ile derin öğrenme arasındaki fark nedir"
]

# =============================================================================
# 3. Model bilgisi ve her yanıt için maliyet (USD)
# =============================================================================
model_info = {
    "gpt-3.5-turbo": {"price_per_prompt": 0.002, "name": "GPT 3.5"},
    "gpt-4": {"price_per_prompt": 0.03, "name": "GPT 4"}
}


# =============================================================================
# 4. Belirli bir prompt için iki farklı modelden cevap üret
# =============================================================================
def generate_two_responses(prompt_text):
    """
    İki farklı modelden (rastgele sıralı) cevap üretip döndürür.
    """
    models = list(model_info.keys())
    random.shuffle(models)

    responses = {}
    for model in models:
        completion = client.chat.completions.create(
            model=model,
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
        content = completion.choices[0].message.content.strip()
        responses[model] = content

    return responses


# =============================================================================
# 5. Kullanıcıdan hangi cevabın daha iyi olduğunu sor
# =============================================================================
def ask_user_to_compare(prompt, responses):
    """
    Kullanıcıdan hangi modelin cevabını tercih ettiğini alır.
    """
    model_a, model_b = list(responses.keys())

    print("\nSoru:")
    print(prompt)
    print("\nBirinci modelin cevabı:")
    print(responses[model_a])
    print("\nİkinci modelin cevabı:")
    print(responses[model_b])

    while True:
        choice = input(
            f"\nHangi cevabı daha iyi buluyorsun? "
            f"{model_info[model_a]['name']} için 1, "
            f"{model_info[model_b]['name']} için 2 yaz: "
        ).strip()

        if choice == "1":
            return model_a, model_b
        elif choice == "2":
            return model_b, model_a

        print("Lütfen sadece 1 ya da 2 giriniz.")


# =============================================================================
# 6. Değerlendirme metrikleri için varsayılan yapı
# =============================================================================
evaluation_results = defaultdict(lambda: {"selected": 0, "cost": 0.0})

# Kaç karşılaştırma yapılacağı
comparison_count = 3

# =============================================================================
# 7. Değerlendirme işlemi
# =============================================================================
for prompt in random.sample(prompt_list, k=comparison_count):
    responses = generate_two_responses(prompt)
    winner_model, loser_model = ask_user_to_compare(prompt, responses)

    # Seçilen modelin kazanç ve maliyetini güncelle
    evaluation_results[winner_model]["selected"] += 1
    evaluation_results[winner_model]["cost"] += \
        model_info[winner_model]["price_per_prompt"]

    # Seçilmeyen model de cevap ürettiği için maliyeti var
    _ = evaluation_results[loser_model]  # defaultdict tetikleme
    evaluation_results[loser_model]["cost"] += \
        model_info[loser_model]["price_per_prompt"]

# =============================================================================
# 8. Sonuçların yazdırılması
# =============================================================================
print("\nModel tercih ve maliyet analizi:")
for model in model_info.keys():  # belirlenen sırayla yaz
    data = evaluation_results[model]
    selected = data["selected"]
    cost = data["cost"]
    name = model_info[model]["name"]

    tercih_basi_maliyet = (
        cost / selected if selected > 0 else float("inf")
    )

    print(
        f"{name} modeli toplam {selected} kez seçildi ve toplam maliyeti "
        f"{cost:.4f} dolar oldu"
    )
    print(
        f"Bir tercih başına ortalama maliyet: "
        f"{tercih_basi_maliyet:.4f} dolar\n"
    )
