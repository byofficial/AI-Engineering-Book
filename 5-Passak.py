import textwrap


# =============================================================================
# 1. Problem Tanımı
# =============================================================================
def display_problem():
    """
    Kullanıcıya problemi açık bir şekilde gösterir.
    """
    print("LeetCode Problemi:")
    print(textwrap.dedent("""
        Bir tamsayı `n` verildiğinde, eğer sayı çiftse `True`,
        tekse `False` döndüren bir fonksiyon yazınız.

        Örnek:
        Girdi: 4  -> Çıktı: True
        Girdi: 7  -> Çıktı: False
    """))


# =============================================================================
# 2. Kullanıcıdan gelen farklı çözüm senaryoları (doğru + hatalı)
# =============================================================================
def get_user_solutions():
    """
    Farklı çözüm stratejileri döndürür.
    Bazıları doğru, bazıları bilinçli olarak hatalıdır.
    """
    return [
        (
            lambda n: n % 2 == 0,
            "Çözüm 1: Doğru kod → `return n % 2 == 0`"
        ),
        (
            lambda n: n % 2 == 1,
            "Çözüm 2: Hatalı kod → `return n % 2 == 1` "
            "(tekliği kontrol ediyor)"
        ),
        (
            lambda n: True,
            "Çözüm 3: Hatalı kod → `return True` (her zaman doğru döndürüyor)"
        ),
        (
            lambda n: "Çift" if n % 2 == 0 else "Tek",
            "Çözüm 4: Yanlış çıktı tipi → `Çift` veya `Tek` döndürüyor, "
            "bool değil"
        ),
        (
            lambda n: 1 / 0,
            "Çözüm 5: Runtime Error → `ZeroDivisionError` fırlatıyor"
        )
    ]


# =============================================================================
# 3. Test Senaryoları
# =============================================================================
def get_test_cases():
    """
    Her çözüm için test edilecek sabit input-output örneklerini verir.
    """
    return [
        (2, True),
        (3, False),
        (10, True),
        (15, False)
    ]


# =============================================================================
# 4. Belirli bir çözüm fonksiyonunu test eden yardımcı fonksiyon
# =============================================================================
def run_test_cases(solution_fn):
    """
    Verilen çözüm fonksiyonu tüm testleri geçiyor mu kontrol eder.
    """
    try:
        for input_val, expected in get_test_cases():
            result = solution_fn(input_val)
            if result != expected:
                return False
        return True
    except Exception:
        return False


# =============================================================================
# 5. Simülasyon: pass@k metriği
# =============================================================================
def simulate_pass_at_k(k=3):
    """
    LeetCode benzeri senaryolarda kullanılan pass@k metriğini simüle eder.
    Yani: k farklı çözümden en az biri tüm testleri geçebiliyor mu?
    """
    display_problem()
    print(f"\n Deneme sayısı (k): {k}\n")

    solutions = get_user_solutions()
    passed_any = False

    for i in range(min(k, len(solutions))):
        fn, description = solutions[i]
        print(f" Deneme {i + 1}: {description}")

        try:
            for input_val, expected in get_test_cases():
                result = fn(input_val)
                status = "✓" if result == expected else "X"
                print(
                    f"   - Test({input_val}) = {result} | "
                    f"Beklenen = {expected} => {status}"
                )
        except Exception as e:
            print(f"   - Hata oluştu: {str(e)}")

        is_pass = run_test_cases(fn)
        if is_pass:
            passed_any = True
            print("   ✓ Bu çözüm TÜM testleri geçti!\n")
        else:
            print("   X Bu çözüm testleri geçemedi.\n")

    # Final karar: pass@k testi başarılı mı?
    print("Final Sonuç:")
    if passed_any:
        print(
            f"✓ Model pass@{k} testini BAŞARDI → "
            "En az bir çözüm tüm testleri geçti."
        )
    else:
        print(
            f"X Model pass@{k} testini BAŞARAMADI → "
            "Hiçbir çözüm tüm testleri geçemedi."
        )


simulate_pass_at_k(k=5)
