import matplotlib.pyplot as plt
import numpy as np

# Rastgelelik için sabit (aynı sonuçları üretir)
np.random.seed(42)

# =============================================================================
# 1. Senaryo Açıklaması: E-Ticaret Müşteri Segmentasyonu
# =============================================================================
# Her müşteri aşağıdaki iki özellik ile temsil edilir:
#  - monthly_spending: Aylık harcama (₺)
#  - last_purchase_gap: Son alışverişten bu yana geçen gün sayısı
#
# 3 farklı müşteri segmenti:
#  Segment 0: VIP Müşteriler (Yüksek harcama, sık alışveriş)
#  Segment 1: Sadık ama az harcayan (Orta harcama, orta sıklık)
#  Segment 2: Uykuda/Kaybedilmiş (Düşük harcama, uzun süredir alışveriş yapmayan)
# =============================================================================

# =============================================================================
# 2. Müşteri Verilerinin Üretilmesi
# =============================================================================

# VIP müşteriler
vip = np.column_stack((
    np.random.normal(15000, 2000, 40),  # Aylık harcama
    np.random.normal(3, 2, 40)  # Gün
))

# Sadık ama daha az harcayanlar
sadik = np.column_stack((
    np.random.normal(3500, 800, 50),
    np.random.normal(15, 5, 50)
))

# Kaybedilmiş veya uykuda olanlar
uykuda = np.column_stack((
    np.random.normal(500, 200, 60),
    np.random.normal(60, 10, 60)
))

# Tüm verileri birleştir
X = np.vstack((vip, sadik, uykuda))

# Segment açıklamaları
segment_desc = {
    0: "VIP Müşteriler: Yüksek harcama, sık alışveriş yapanlar",
    1: "Sadık Müşteriler: Orta harcama, düzenli alışveriş yapanlar",
    2: "Uykuda/Kaybedilmiş: Uzun süredir alışveriş yapmayan, düşük harcamalı"
}


# =============================================================================
# 3. K-Means Algoritması
# =============================================================================
def realistic_kmeans(X, k=3, max_iters=10, delay=2):
    """
    K-Means algoritmasını görselleştirerek çalıştırır.

    Parametreler:
    -------------
    X : np.ndarray
        Veri kümesi (müşteri özellikleri)
    k : int
        Küme sayısı
    max_iters : int
        Maksimum iterasyon sayısı
    delay : int or float
        Her iterasyondan sonra kaç saniye beklenilecek (görselleştirme için)

    Açıklama:
    ---------
    1. Rastgele k merkez seçilir.
    2. Her müşteri, en yakın merkeze atanır.
    3. Merkezler, ait oldukları müşterilerin ortalaması olarak güncellenir.
    4. Merkezler değişmeyene kadar devam eder.
    """
    n_samples, n_features = X.shape

    # 1. Başlangıç merkezlerini rastgele seç
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for i in range(max_iters):
        # 2. Tüm müşteriler ile merkezler arası Öklid uzaklığı
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # 3. Her müşteri en yakın merkeze atanır (küme belirlenir)
        clusters = np.argmin(distances, axis=1)

        # 4. Görselleştirme
        plt.figure(figsize=(10, 6))
        plt.title(f"K-Means - Iterasyon {i + 1}")
        plt.xlabel("Aylık Harcama (₺)")
        plt.ylabel("Son Alışverişten Geçen Gün")
        plt.grid(True)

        for j in range(k):
            cluster_points = X[clusters == j]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        label=f"Segment {j}")

        # Küme merkezlerini siyah 'X' ile göster
        for idx, (cx, cy) in enumerate(centroids):
            plt.scatter(cx, cy, c='black', marker='X', s=250)
            plt.text(cx + 300, cy + 1,
                     f"Merkez {idx}\n({cx:.0f}₺, {cy:.1f} gün)",
                     fontsize=9, color='black')

        plt.legend()
        plt.pause(delay)
        plt.show()

        # 5. Yeni merkezleri güncelle
        new_centroids = np.array([
            X[clusters == j].mean(axis=0) for j in range(k)
        ])

        # 6. Merkezlerde değişim yoksa dur
        if np.allclose(centroids, new_centroids):
            print(f"\nMerkezler sabitlendi. {i + 1}. iterasyonda durdu.")
            break

        centroids = new_centroids

    # =============================================================================
    # 4. Sonuçların Raporlanması
    # =============================================================================
    print("\nFinal Segment Özeti:")
    for idx, (cx, cy) in enumerate(centroids):
        print(f"\nSegment {idx}")
        print(f"  → Ortalama Harcama: ₺{cx:,.0f}")
        print(f"  → Ortalama Son Alışveriş Gün Farkı: {cy:.1f} gün")
        print(f"  → Açıklama: {segment_desc.get(idx, 'Tanımsız')}")
        print("-" * 40)


realistic_kmeans(X, k=3, max_iters=10, delay=2)
