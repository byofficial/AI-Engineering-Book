import matplotlib.pyplot as plt
import numpy as np

# 1. Veri üretimi
# 3 farklı küme etrafında normal dağılımlı 2 boyutlu veri oluşturuluyor.
np.random.seed(0)
X = np.vstack((
    np.random.normal([2, 2], 0.5, (50, 2)),
    np.random.normal([6, 6], 0.5, (50, 2)),
    np.random.normal([10, 2], 0.5, (50, 2))
))


def kmeans_with_slow_visuals(X, k, max_iters=10, delay=2):
    """
    K-Means algoritmasını adım adım ve görselleştirerek çalıştırır.

    Parametreler:
    ------------
    X : np.ndarray
        Giriş veri kümesi, her satır bir veri noktasıdır.
    k : int
        Küme (cluster) sayısı.
    max_iters : int, optional
        Maksimum iterasyon sayısı (default=10).
    delay : int or float, optional
        Her iterasyon arasında bekleme süresi (saniye olarak).

    Yöntem:
    -------
    1. Rastgele k merkez seçilir.
    2. Her veri noktası, en yakın merkeze atanır.
    3. Merkezler, ait oldukları noktaların ortalaması olarak güncellenir.
    4. Merkezler değişmeyene kadar ya da max iterasyon dolana kadar devam eder.
    """
    n_samples, n_features = X.shape

    # Rastgele k tane başlangıç merkezi seçilir
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for i in range(max_iters):
        # 1. Her nokta ile her merkezin uzaklığı hesaplanır
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # En yakın merkeze ait indeks (küme) belirlenir
        clusters = np.argmin(distances, axis=1)

        # Görselleştirme başlatılır
        plt.figure(figsize=(8, 6))
        plt.title(f"İterasyon {i + 1}")
        plt.grid(True)

        # Her kümedeki noktalar farklı renk ile çizilir
        for j in range(k):
            cluster_points = X[clusters == j]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        label=f"Küme {j}")

        # Mevcut merkezler siyah X işareti ile gösterilir
        for idx, (cx, cy) in enumerate(centroids):
            plt.scatter(cx, cy, c='black', marker='X', s=250)
            plt.text(cx + 0.1, cy + 0.1,
                     f"Merkez {idx}\n({cx:.2f}, {cy:.2f})",
                     fontsize=9, color='black')

        plt.legend()
        plt.pause(delay)
        plt.show()

        # 2. Yeni merkezler hesaplanır (ait olduğu noktaların ortalaması)
        new_centroids = np.array([
            X[clusters == j].mean(axis=0) for j in range(k)
        ])

        # Eğer merkezlerde değişiklik yoksa algoritma sonlandırılır
        if np.allclose(centroids, new_centroids):
            print(f"Küme merkezleri sabitlendi. "
                  f"{i + 1}. iterasyonda durdu.")
            break

        # Yeni merkezler ile devam edilir
        centroids = new_centroids

    # Son merkez koordinatları yazdırılır
    print("Son merkez koordinatları:")
    for idx, (cx, cy) in enumerate(centroids):
        print(f"Merkez {idx}: ({cx:.2f}, {cy:.2f})")


# K-Means algoritması 3 küme için çalıştırılır
kmeans_with_slow_visuals(X, k=3, max_iters=10, delay=2)
