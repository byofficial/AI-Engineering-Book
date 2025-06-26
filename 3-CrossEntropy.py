import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

"""
 Cross-Entropy Açıklaması:
Gerçek değer y = 1, tahmin edilen olasılık y_hat = 0.9 ise:

Cross-entropy = - [1 * log(0.9) + 0 * log(0.1)]
               = - log(0.9)
               ≈ 0.105

Cross-entropy, doğru sınıfa ne kadar "kesin" tahminde bulunduğunu ölçer.
Küçük değer daha iyi performans gösterir.

 Perplexity:
Cross-entropy'nin üsse çevrilmiş halidir.
Belirsizlik ölçüsüdür. Perplexity ne kadar düşükse, model o kadar iyi.
"""

# ============================================================================
# 1. Veriyi oku
# ============================================================================
df = pd.read_csv("data/bank-additional-full.csv", sep=';')

# ============================================================================
# 2. Hedef sütunu dönüştür (binary): 'yes' → 1, 'no' → 0
# ============================================================================
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# ============================================================================
# 3. Kategorik ve sayısal sütunları ayır
# ============================================================================
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('y')  # hedef sütun analizden çıkarıldı

# ============================================================================
# 4. Kategorik değişkenleri One-Hot Encode et
# ============================================================================
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(df[cat_cols])

# ============================================================================
# 5. Sayısal verileri normalize et (ortalama=0, std=1)
# ============================================================================
scaler = StandardScaler()
X_num = scaler.fit_transform(df[num_cols])

# ============================================================================
# 6. Tüm özellikleri birleştir
# ============================================================================
X = np.hstack((X_num, X_cat))
y = df['y'].values

# ============================================================================
# 7. Eğitim/Test bölmesi (%80 eğitim, %20 test)
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================================
# 8. Lojistik Regresyon modelini oluştur ve eğit
# ============================================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================================================================
# 9. Tahmin yap
# ============================================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # log loss için olasılıklar

# ============================================================================
# 10. Model değerlendirmesi
# ============================================================================
acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_proba)

# ============================================================================
# 11. Perplexity (belirsizlik ölçütü)
# ============================================================================
perplexity = math.exp(loss)  # e^cross-entropy

# ============================================================================
# 12. Sonuçları yazdır
# ============================================================================
print("Accuracy (Kesinlik):", round(acc * 100, 2), "%")
print("Cross-Entropy (Log Loss):", round(loss, 4))
print("Perplexity:", round(perplexity, 4))
