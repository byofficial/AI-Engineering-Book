import os
import re
import textwrap

from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasından API anahtarını yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# 1. Problem Tanımı
# =============================================================================
PROBLEM_DESCRIPTION = textwrap.dedent("""
    Bir tamsayı `n` verildiğinde, eğer sayı çiftse `True`, 
    tekse `False` döndüren bir Python fonksiyonu yazınız.

    Örnek:
    Girdi: 4  -> Çıktı: True
    Girdi: 7  -> Çıktı: False

    Sadece fonksiyonu döndür. Açıklama ekleme. 
    Farklı yollarla çözümler üretmeye çalış.
""")

# =============================================================================
# 2. Test Senaryoları
# =============================================================================
TEST_CASES = [
    (2, True),
    (3, False),
    (10, True),
    (15, False)
]


# =============================================================================
# 3. Problemi ekrana yazdıran yardımcı fonksiyon
# =============================================================================
def display_problem():
    print("LeetCode Problemi:")
    print(PROBLEM_DESCRIPTION)


# =============================================================================
# 4. GPT-4 ile çözüm oluşturma
# =============================================================================
def generate_solution_with_gpt(index):
    """
    GPT-4'e problem açıklamasını vererek farklı yöntemle bir çözüm üretmesini sağlar.
    """
    print(f"\n GPT-4 ile çözüm {index + 1} oluşturuluyor...")

    prompt = PROBLEM_DESCRIPTION + (
        f"\n\nNot: Bu {index + 1}. farklı çözüm olsun. "
        "Her seferinde farklı bir yöntemle dene."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Sen yetenekli bir Python geliştiricisisin."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1.4,
        top_p=0.95,
        frequency_penalty=0.6,
        presence_penalty=0.4,
        max_tokens=200
    )

    code = response.choices[0].message.content
    print(f"GPT-4 Çözüm {index + 1}:\n{code}")
    return extract_code_block(code)


# =============================================================================
# 5. Kod bloğunu cevaptan ayıklayan yardımcı fonksiyon
# =============================================================================
def extract_code_block(response_text):
    """
    Triple backtick içinde gelen kod bloğunu ayrıştırır.
    """
    code_blocks = re.findall(
        r"```(?:python)?\n(.*?)```", response_text, re.DOTALL
    )
    return code_blocks[0].strip() if code_blocks else response_text.strip()


# =============================================================================
# 6. Üretilen kodu çalıştır ve test et
# =============================================================================
def safe_execute(code_string):
    """
    exec ile dinamik kod çalıştırır ve test senaryolarına göre değerlendirir.
    """
    try:
        local_env = {}
        exec(code_string, {}, local_env)
        fn = next((v for v in local_env.values() if callable(v)), None)

        if fn is None:
            return False, "Fonksiyon bulunamadı."

        for input_val, expected in TEST_CASES:
            result = fn(input_val)
            if result != expected:
                return False, (
                    f"Test({input_val}) = {result}, "
                    f"Beklenen = {expected}"
                )

        return True, "Tüm testler geçti."

    except Exception as e:
        return False, f"Hata: {str(e)}"


# =============================================================================
# 7. pass@k simülasyonu
# =============================================================================
def simulate_pass_at_k(k=5):
    """
    GPT-4'ün k farklı çözümünden en az birinin testleri geçip geçmediğini kontrol eder.
    """
    display_problem()
    print(f"\n GPT-4 ile {k} farklı çözüm denenecek...\n")

    passed = False

    for i in range(k):
        code = generate_solution_with_gpt(i)
        result, detail = safe_execute(code)

        if result:
            passed = True
            print(f"✓ Deneme {i + 1} → BAŞARILI: {detail}")
        else:
            print(f"X Deneme {i + 1} → BAŞARISIZ: {detail}")

    print("\n Final Sonuç:")
    if passed:
        print(
            f"✓ GPT-4 pass@{k} testini BAŞARDI → "
            "En az bir çözüm tüm testleri geçti."
        )
    else:
        print(
            f"X GPT-4 pass@{k} testini BAŞARAMADI → "
            "Hiçbir çözüm tüm testleri geçemedi."
        )


simulate_pass_at_k(k=5)
