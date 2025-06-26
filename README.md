# Chapter 3. Evaluation Methodology (Değerlendirme Metodolojisi)

- AI kullanımı arttıkça **kritik hatalar riski** büyüyor.
- AI projelerinde en zorlayıcı adım çoğu zaman **“değerlendirme”** oluyor.
- Değerlendirme ayrı bir adım değil, **sistemin genel yapısıyla entegre düşünülmeli**.
- Temel modelleri değerlendirmek zordur çünkü **çıktılar açık uçludur**, yani çok sayıda geçerli yanıt olabilir.
- Değerlendirme için insanlar kullanılsa da **otomasyon** şarttır.
- **AI’ın AI’ı değerlendirmesi** (AI as a judge) giderek popülerleşiyor, ama tartışmalı bir konu.
- Bu bölümde:
    - Kullanılan **değerlendirme yöntemleri**,
        - **Cross-entropy**, **perplexity** gibi metrikler,
        - **Açık uçlu yanıtları değerlendirme** stratejileri anlatılmaktadır.

## Foundation Modeller Nedir?

**Foundation models** (temel modeller), çok büyük veri setleri üzerinde **genel amaçlı olarak eğitilmiş**, **çok
çeşitli  
görevlerde** kullanılabilen yapay zeka modelleridir.

Bu modeller, sonradan **özelleştirilebilir**, yani başka görevler için **ince ayar (fine-tuning)** yapılabilir.

### **Özellikleri:**

- Çok **geniş veri setleri** (milyarlarca token) üzerinde eğitilirler.
- Genellikle **kendinden denetimli (self-supervised)** öğrenme teknikleriyle eğitilirler.
- **Doğal dil işleme**, **görsel algılama**, **kod yazımı**, **özetleme**, **çeviri**, hatta **robot hareketi planlama
  **  
  gibi birçok alanda kullanılabilirler.

### Neden “foundation” (temel) deniyor?

Çünkü bu modeller, farklı uygulamalar için bir **altyapı (temel)** görevi görür. Örneğin:

- GPT modelleri (ChatGPT’nin temelini oluşturan),
- BERT, RoBERTa gibi transformer tabanlı dil modelleri,
- CLIP, DALL·E, Stable Diffusion gibi çok modalite destekleyen modeller

bunların hepsi foundation model örneğidir.

# Algoritmalar

## K-Means

**K-Means**, denetimsiz öğrenme (unsupervised learning) yöntemlerinden biridir ve temel amacı:

**Veri noktalarını benzerliklerine göre gruplamak (clustering)**

### **Nerede kullanılır?**

- Müşteri segmentasyonu (örneğin bankacılıkta “benzer harcama yapan müşteriler”)
- Görüntü segmentasyonu (aynı renge veya yapıya sahip pikselleri grupla)
- Pazarlama kampanya hedefleme
- Web sayfası ziyaretçi gruplama

### Nasıl çalışır?

1. **K (küme sayısı)** belirlenir (önceden verilmelidir)
2. Rastgele **K adet merkez (centroid)** seçilir
3. Her veri noktası, en yakın merkeze atanır (Öklid uzaklığı ile)
4. Her küme için yeni merkezler, kümedeki noktaların **ortalaması** alınarak güncellenir
5. Atama ve güncelleme adımları **merkezler değişmeyene kadar** tekrarlanır

### **Dezavantajları**

- K’yı **önceden bilmek gerekir**
- Küme sayısı yanlışsa gruplama kötü olur
- Küresel olmayan (dağınık, uzamış) kümeleri **kötü yakalar**
- Hatalı başlangıç → kötü sonuç
- Gürültüye ve aykırı değerlere duyarlıdır

## Cross-Entropy

**Cross-entropy**, bir modelin yaptığı tahminin **gerçek etiketle ne kadar uyuşmadığını ölçen** bir **kayıp (loss)  
fonksiyonudur**.

> “Gerçeği ne kadar iyi tahmin ettin?”

> Tahminlerin ne kadar uzaksa, cross-entropy o kadar büyüktür.

## **Nerede Kullanılır?**

- **Binary sınıflandırma** (örnek: e-posta spam mi, değil mi?)
- **Çok sınıflı sınıflandırma** (örnek: görüntüdeki nesne kedi/köpek/kuş mu?)
- **Dil modelleri (ChatGPT, BERT)** → sıradaki kelime tahmini
- **Sinir ağları, lojistik regresyon**, CNN, RNN, LSTM

## Perplexity

**Perplexity**, bir dil modelinin (veya herhangi bir olasılık modeli) ne kadar “şaşırdığını” gösterir.

Yani:

> Model, test verisini  
> **ne kadar iyi tahmin edebiliyor?**

Daha matematiksel bir anlatımla:

> Bir modelin bir dizi kelimeyi üretme olasılığı ne kadar yüksekse,  
> **perplexity o kadar düşük olur.**

> Bu da modelin o diziyi “tahmin etmede iyi olduğunu” gösterir.

- Perplexity = 1 → Model hiç şaşırmıyor, yani mükemmel tahmin yapıyor.
- Perplexity = 10 → Ortalama 10 farklı seçenek arasında kararsız kalıyor.
- Perplexity ↑ yüksekse → Modelin belirsizliği ↑ demektir.

### **Nerede Kullanılır?**

- GPT, BERT gibi dil modellerinin eğitim ve testinde
- Otomatik konuşma tanıma sistemlerinde
- Makine çeviri modellerinde
- Text generation görevlerinde (örneğin: otomatik e-posta üretimi)

# Understanding Language Modeling Metrics

## **Zeka Arttıkça Değerlendirme Zorlaşıyor**

- Basit bir sınıflandırma modelinde yanlış cevap kolay fark edilir.
- Ama bir doktora seviyesi matematik çözümünü doğru mu yanlış mı ayırt etmek için uzmanlık gerekir.
- Örneğin, bir özetin doğru olup olmadığını anlamak için kitabı baştan sona okuyup özetle karşılaştırmak gerekir.

**Zorluk:** İnsan benzeri yanıtlar üreten modelleri değerlendirmek uzmanlık, zaman ve domain bilgisi ister.

## **Açık-Uçlu Cevaplar = Zor Karar**

- Geleneksel ML’de “ground truth” bellidir: Sınıf A mı B mi?
- Ama LLM gibi modeller “açık uçlu” çalışır. Cevaplar çok çeşitlidir ve hepsi doğru olabilir.

**Örnek:** “Türkiye ekonomisi hakkında ne düşünüyorsun?” → 10 farklı model 10 farklı yanıt verebilir ama hepsi
mantıklı  
olabilir.

## **Model Şeffaflığı Yok (Black Box)**

- Çoğu foundation modelin iç yapısı bilinmez.
- Ne veriyle eğitildi, hangi mimariyle geliştirildi, bilinmiyor.

**Sonuç:** Yalnızca modelin çıktısına bakarak değerlendirme yapılabiliyor. Bu da yüzeysel kalabilir.

## **Benchmark’lar Hızla Eskidi**

- Örneğin GLUE → 2018’de çıktı, 1 yılda doygunluğa ulaştı.
- Yerine SuperGLUE geldi. Şimdi MMLU-Pro, Super-NaturalInstructions gibi yeni nesil benchmark’lar gerekiyor.

**Sebep:** Modeller hızla gelişiyor, eski benchmark’lar artık ayırt edici değil.
  
-----  

# Exact Evaluation (Kesin Değerlendirme)

Model performanslarını değerlendirirken, **kesin (exact)** ve **öznel (subjective)** değerlendirme arasında ayrım yapmak
önemlidir.

- **Kesin değerlendirme**, yoruma açık olmayan net bir yargı sunar.

  Örneğin:

  > Çoktan seçmeli bir sorunun cevabı A ise ve biz B’yi seçtiysek, bu yanlıştır. Burada herhangi bir belirsizlik yoktur.

- **Öznel değerlendirme** ise kişisel yoruma dayanır.

  Örneğin:

  > Bir kompozisyonun puanı, kim tarafından değerlendirildiğine göre değişebilir. Aynı kişi, aynı yazıya farklı
  zamanlarda farklı puan verebilir.

Kompozisyon değerlendirmesi, açık ve net yönergelerle daha “kesin” hale getirilebilir.

## **Functional Correctness (Fonksiyonel Doğruluk)**

Bir sistemin istenen işlevi gerçekten yerine getirip getirmediğine bakılır.

Örnekler:

- Bir modele “web sitesi yap” dedik → **İstediğimiz siteyi yaptı mı?**
- “Şu restorana rezervasyon yap” dedik → **Rezervasyon gerçekleşti mi?**

Bu, **bir uygulamanın işini yapıp yapmadığını gösteren en anlamlı metrik**tir.

Ama her zaman ölçülmesi kolay değildir ve genellikle **otomatikleştirilemez**.

## **Similarity Measurements Against Reference Data (Referans Veriye Göre Benzerlik Ölçümü)**

- Eğer fonksiyonel doğruluk otomatik ölçülemiyorsa, **çıktılar referans (ground truth) verilerle karşılaştırılır.**
- Örn: Fransızca’dan İngilizce’ye çeviri → AI çıktısı referans çeviriyle karşılaştırılır.
- Her örnek şu şekilde olur: (input, reference responses)
    - Bir input birden fazla referansa sahip olabilir.
- **Referanslı metrikler** (reference-based) ve **referanssız metrikler** (reference-free) ayrımı yapılır.
- **İnsan üretimi referans** → Altın standarttır, ama zaman ve maliyetlidir.
- AI ile referans üretimi daha hızlıdır ama insan kontrolü gerekebilir.
- **Benzerlik ölçüm yöntemleri:**

1. **Exact match**
2. **Lexical similarity**
3. **Semantic similarity**

## **Exact Match (Birebir Eşleşme)**

- Model çıktısı, referanslardan biriyle **kelimesi kelimesine aynıysa başarılı** sayılır.
- Kısa cevaplı görevlerde uygundur:
    - “2 + 3 nedir?” → “5”
    - “İlk Nobel kazanan kadın kimdir?”
- Bazı varyantlar küçük biçim farklılıklarına tolerans tanır:
    - “5” yerine “Sonuç 5’tir” de kabul edilebilir.
- Ancak bu tolerans, yanlış sonuçların kabul edilmesine de yol açabilir.
- Karmaşık görevlerde **yetersiz kalır** çünkü:
    - Bir input için sonsuz doğru ifade şekli olabilir.
    - Örn: “Comment ça va?” → “How is it going?” yoksa yanlış sayılır.

## **Lexical Similarity (Leksik Benzerlik)**

- Metinler arasındaki **yüzeysel (kelime düzeyinde) benzerliği** ölçer.
- Yöntemler:
    - **Token overlap (kelime eşleşmesi)**: Ortak kelime yüzdesi
    - **Fuzzy matching (yaklaşık eşleşme)**: Edit distance (ekleme, silme, değiştirme)
    - **n-gram overlap**: “my cats scare the mice” → bigram’larla ölçüm
- **Ölçüm metrikleri**:
    - BLEU, ROUGE, METEOR, TER, CIDEr
- **Eksileri:**
- Referans eksikse, doğru cevap düşük puan alabilir.
- Referans hatalı olabilir → WMT 2023’te kötü çeviriler tespit edildi.
- Yüksek skorlar her zaman doğruya işaret etmez (BLEU ile yanlış kodlar da yüksek puan alabilir).

## **Semantic Similarity (Anlamsal Benzerlik)**

- Metinlerin **anlam bakımından ne kadar benzer olduğunu** ölçer.
- Gereken ilk adım: Metni sayısal vektöre dönüştürmek → **embedding**
- Örn: “the cat sits on a mat” → [0.11, 0.02, 0.54]
- En yaygın ölçüm: **Cosine similarity**
- Aynı anlamlı embedding’ler → skor = 1
- Zıt embedding’ler → skor = –1
- **Popüler metrikler:**
- **BERTScore** (BERT tabanlı)
    - **MoverScore**
- **Avantajları:**
- Leksik eşleşme kadar çok referans gerekmez.
- **Dezavantajları:**
- Embedding algoritmasının kalitesine bağlıdır.
- Hesaplama açısından daha maliyetlidir.
- Ses, görsel vb. diğer modaliteler için de kullanılabilir.

## **Embedding’e Giriş (Introduction to Embedding)**

- Embedding = metnin **anlamını temsil eden sayısal vektör**.
- Tipik boyut: 100–10.000 arası.
- Yaygın modeller:
    - **BERT** → 768–1024 boyut
    - **CLIP** (text/image): 512
    - **OpenAI text-embedding-3**: 1536–3072
    - **Cohere Embed v3**: 384–1024
- Embedding’lerin kalitesi şunlarla ölçülür:
    - Benzer cümlelerin embedding’leri birbirine yakın mı?
    - Görev performansı: sınıflandırma, RAG, arama, öneri, vs.
- **Benchmark örneği:** MTEB
- **Multimodal embedding (çoklu veri türü)**:
    - CLIP → text + image
    - ULIP → text + image + 3D
    - ImageBind → 6 farklı modalite
- **Kullanım:** Arama, eşleştirme, açıklama üretimi, çok modaliteli arama, vs.

# AI as a Judge

Açık uçlu (open-ended) cevapların değerlendirilmesindeki zorluklardan yola çıkarak, AI modellerinin kendilerini veya
birbirlerini değerlendirmek için nasıl kullanılabileceğini anlatıyor. Bu yaklaşıma **“AI as a Judge”** (hakem olarak
yapay zeka) ya da **“LLM as a Judge”** deniyor. Değerlendirme yapan modele **AI Judge** (yapay zekâ hakemi) adı
veriliyor.

Bu fikir çok eski olsa da, ancak **GPT-3’ün 2020’deki çıkışıyla** pratikte uygulanabilir hale geldi. 2023 ve 2024
yıllarında yapılan değerlendirmelerde artık **en çok tercih edilen yöntemlerden biri** haline geldi.

LangChain’in 2023 AI raporuna göre, platformlarındaki değerlendirmelerin **%58’i AI judges** tarafından yapıldı. Yani
üretimde yaygın ve aktif bir araştırma alanı.

## **Neden AI Hakemi Kullanılır?**

### Avantajları:

- Hızlı,
- Kullanımı kolay,
- İnsan değerlendiricilere kıyasla çok daha ucuz,
- Referans verisi olmadan da çalışabilir.

> Bu, üretim ortamlarında büyük avantaj çünkü her zaman “doğru cevap” elimizde olmaz.

### Esneklik:

AI modellerine aşağıdaki gibi her türlü kriter için değerlendirme yaptırabiliriz:

- Doğruluk (correctness)

- Tekrarlılık (repetitiveness)

- Zehirlilik (toxicity)

- Gerçeklikten sapma (hallucination)

- Uygunluk (wholesomeness)

  ve daha fazlası…

> İnsanlardan “bir fikrini” söylemesini istemek gibi… Her zaman güvenemeyiz ama çoğunluğun eğilimini yansıtır.

## How to Use AI as a Judge

Bu bölüm, **AI as a Judge (Hakem olarak AI)** yaklaşımının **nasıl uygulanacağına** dair pratik ve teknik yolları
anlatıyor.

Temel olarak üç ana değerlendirme şekli tanıtılıyor:

### **Cevabın kendi başına değerlendirilmesi**

> “Bu cevap iyi mi?” tarzında.

- Sadece soru ve cevaba bakarak kalite puanı verilir (örneğin 1–5 arasında).
- Bu, **en basit ve yaygın** değerlendirme senaryosudur.

----------  

### **Cevabın referans cevaba göre değerlendirilmesi**

> “Bu cevap, referans cevapla aynı mı?”

- “True/False” gibi ikili sınıflandırmalar kullanılır.
- BLEU, ROUGE gibi metrikler yerine LLM ile yapılır (örn. “human-like comparison”).

----------  

### **İki cevabın karşılaştırılması (Preference Evaluation)**

> “Hangisi daha iyi?”

Bu, özellikle:

- Model sıralaması (A vs B),
- Post-training alignment (RLHF),
- Test-time seçim ve reranking gibi konularda kullanılır.

----------  

## Limitations of AI as a Judge

AI judge kullanımı değerlendirmenin maliyetini, ölçeklenebilirliğini ve hızını artırsa da bazı temel problemler ve
sınırlamalar içeriyor. Bu nedenle bazı takımlar AI değerlendirmeyi ancak başka çareleri olmadığında tercih ediyor.

Kitap burada çok dengeli bir eleştiri sunuyor: _AI yargıç faydalı olabilir ama “tek başına güvenilemez.”_

### **Inconsistency (Tutarsızlık)**

- AI sistemleri deterministik değil, _probabilistic_ çalışır.

  → Aynı girdiye aynı promptla bile farklı çıktılar dönebilir.

- Bu, değerlendirme sonuçlarının **tekrar edilebilirliğini** düşürür.

- **Zheng et al. (2023)**: Prompt içine örnekler koyunca GPT-4’ün tutarlılığı %65’ten %77.5’e çıktı.

    - Ama dikkat: **Tutarlılık ≠ Doğruluk**.
    - Sürekli aynı hatayı yapmak da tutarlılık olabilir.

Tutarlılığı artırmak için prompt uzarsa inference maliyeti ciddi artar. Örnek: GPT-4 harcaması 4 katına çıkmış.
  
----------  

### **Criteria Ambiguity (Kriter Belirsizliği)**

- İnsan eliyle tanımlı metrikler genelde standartken, AI yargıçlarının kriterleri çok farklı olabilir.
- Aynı “faithfulness” metriğini 3 farklı araç farklı şekilde tanımlamış:
    - **MLflow:** 1–5 arası puan
    - **Ragas:** 0 veya 1
    - **LlamaIndex:** YES / NO

➡ Aynı örneğe MLflow 3, Ragas 1, LlamaIndex NO diyebilir.

Bu tür belirsizlikler metrik karşılaştırmasını imkânsız hale getiriyor. Değerlendirme modeli/prompt değişirse aynı
çıktının puanı değişebilir.
  
----------  

### **Evaluation Drift (Zamanla Bozulma)**

- Uygulamanın versiyonu değiştikçe **evaluation kriterleri değişmemeli** ki sonuçlar izlenebilsin.
- Ama AI judge da bir AI olduğu için kendisi de değişebilir:
    - Model güncellenir
    - Prompt ufakça değiştirilir (typo düzeltme bile fark yaratabilir)
    - Bu da skorun değişmesine neden olur.

Özellikle farklı ekipler çalışıyorsa, **AI judge prompt değişikliğini haber vermemiş olabilir.**

Bu durumda uygulama ekibi yanlışlıkla kendi sistemini suçlayabilir.

> Eğer prompt ve model açıkça görülmüyorsa AI judge’a güvenmemeliyiz.
  
----------  

### **Increased Costs and Latency (Maliyet ve Gecikme)**

- Eğer hem üretimi hem değerlendirmeyi GPT-4 ile yaparsak, çağrı sayısı 2 katına çıkar.
    - 3 farklı kriteri ölçersek 4 katına çıkar.
- Özellikle üretimde AI judge kullanılırsa → **API çağrısı + gecikme riski.**

**Spot-checking** çözüm olabilir (bazı örnekleri seçerek değerlendirme).

- Ucuz ama her hatayı yakalamayabilir.
- Daha fazla örnek incelersen maliyet artar, ama güven de artar.

----------  

### **Ne Yapmalı?**

- AI judge kullanacaksak:
    - Prompt ve modeli mutlaka loglamalıyız
    - Spot-checking veya fallback mekanizmaları eklemeliyiz
    - Metrikleri standartlaştırmalı ve sabit tutmalıyız
- Özellikle bias ve tutarsızlığa karşı test setlerini karıştırarak kullanmalıyız

## What Models Can Act as Judges?

Bu bölümde, bir yapay zekâ modelinin “yargıç” rolünde kullanılabileceği senaryolar tartışılıyor. Yani hangi model(ler)
başka bir modelin çıktısını değerlendirebilir?  
Örneğin:

- Daha güçlü model,
- Daha zayıf model,
- Aynı model (self-evaluation)

### **Stronger model as a judge**

- İlk bakışta en mantıklı olan bu gibi görünüyor. Çünkü sınavı değerlendiren kişi, sınavı çözen kişiden daha bilgili
  olmalı.
- Güçlü model, zayıf modellerin çıktısını hem değerlendirebilir hem de onları iyileştirebilir.
- Ancak bu model **daha pahalı** ve **daha yavaş** olabilir. Bu nedenle sadece %1 gibi küçük bir kısmı değerlendirmek
  için kullanılabilir.
- Örnek: Ucuz bir in-house model çıktı üretir, GPT-4 ise sadece örneklem üzerinden değerlendirme yapar.

### **Weaker model as a judge**

- Bazı uzmanlara göre, yargılamak üretmekten daha kolaydır. Yani her birey şarkı yazamaz ama iyi olup olmadığını
  söyleyebilir.
- Daha zayıf bir model, daha güçlü bir modelin çıktısını **değerlendirebilir**, özellikle eğer değerlendirme alanı
  sınırlıysa (örn. grammar, formatting, toxicity).

### **Self-evaluation (Aynı model)**

- Aynı modelin kendi çıktısını değerlendirmesi genelde güvenilmez gibi görülür (self-bias etkisi).
- Ama “sanity check” olarak faydalıdır: Eğer bir model kendi cevabının yanlış olduğunu söyleyebiliyorsa, bu modele
  kısmen güvenilebilir.
- Ayrıca bu yöntem **modeli hatasını fark ettirip düzeltmesini sağlamak için** de kullanılabilir (Press et al., 2022;  
  Gou et al., 2023).

## **Özel Amaçlı AI Hakem Türleri**

### **Reward Model**

- (Prompt, response) çiftini alır, skorlama yapar.
- Genellikle reinforcement learning (RLHF) içinde kullanılır.
- Örnek: Google’ın geliştirdiği **Cappy** modeli — 360 milyon parametreli, çok hafif bir reward scorer.

### **Reference-Based Judge**

- Üretilen cevabı referans cevaba göre değerlendirir.
- Örnekler:
    - **BLEURT** (Sellam et al., 2020): (Candidate, reference) verip benzerlik skoru döner.
    - **Prometheus** (Kim et al., 2023): (prompt, response, reference, rubric) alır; kalite skoru döner.

### **Preference Model**

- (prompt, response1, response2) alır ve hangisinin tercih edildiğini söyler.
- İnsan tercihini modellemek için kullanılır.
- Örnekler:
    - **PandaLM** (Wang et al., 2023)
    - **JudgeLM** (Zhu et al., 2023)

Bu modeller özellikle _insan beğenisi modelleme_, _chatbot seçim_, _multi-turn generation_ gibi yerlerde kullanılır.

# Ranking Models with Comparative Evaluation

Bu bölüm, modellerin **sadece puanlarını değil, hangisinin daha iyi olduğunu öğrenmek** istediğimiz durumlarda nasıl
sıralandığını açıklıyor. İki temel yöntem sunuluyor:

### **Pointwise Evaluation (Noktasal / Bağımsız Değerlendirme):**

- Her model tek başına değerlendirilir.
- Örnek: Her dansçıya ayrı puan verilir, en yüksek puanı alan kazanır.

### **Comparative Evaluation (Karşılaştırmalı Değerlendirme):**

- Modeller birbirlerine karşı değerlendirilir.
- Örnek: Dansçılar yan yana dans eder, hangisi daha çok beğenilirse o kazanır.
- **Özellikle öznel kalite kriterlerinde** daha uygundur. (Müzik, sohbet, yaratıcı yazı gibi)

## Challenges of Comparative Evaluation

### **Scalability Bottlenecks**

- Karşılaştırmalı değerlendirme (comparative evaluation), çok fazla veri ister.
- Model sayısı arttıkça, karşılaştırma yapılması gereken model çiftleri **kare (n²)** hızda artar.
- 57 model için 1.596 model çifti oluşur, ve 244.000 karşılaştırma yapılsa bile **her çift ortalama sadece 153 kez
  karşılaştırılmış** olur. Bu, detaylı sıralama için azdır.
- **Transitivite (geçişlilik)** varsayımı kullanılarak bu yük azaltılabilir: A > B ve B > C ise, A > C denebilir.
- Ancak insan tercihleri geçişli olmayabilir, bu da sıralamanın tutarsız olmasına yol açar.

### ** Dikkat edilmesi gerekenler:**

- Eğer kendi sistemimizde 10’dan fazla model karşılaştırmayı düşünüyorsak, transitive ilişkilere dikkat etmeliyiz.
- Her yeni modelin sisteme girmesi tüm sıralamayı değiştirebilir.
- Yeni modelleri karşılaştırmak için **özel eşleştirme algoritmaları** kullanılabilir (örneğin: daha belirsiz model
  çiftlerine öncelik vermek gibi).

----------  

### **Lack of Standardization and Quality Control**

- Topluluk tabanlı karşılaştırma (ör. LMSYS Chatbot Arena) esneklik sağlar ama kalite sorunludur.
- **Kimin hangi kritere göre oy verdiği belli değildir.**
- Örneğin: modelin toksik cevabı kullanıcı tarafından “eğlenceli” bulunduğu için seçilmesi sıralamayı bozabilir.
- Kullanıcılar da bazen anlamsız, fazla basit (“hello”) veya gerçek kullanım senaryosunu yansıtmayan promptlar kullanır.
- Bazı modeller bağlam oluşturma (retrieval + generation) konusunda iyidir ama leaderboard sadece cevaba bakar. Bu farkı
  göremez.

### ** Dikkat edilmesi gerekenler:**

- Kendi sistemimizde comparative evaluation kullanacaksak:
    - Ya belirli hazır prompt seti tanımlamalıyız (zor ama düzenli),
    - Ya da **prompta göre ağırlıklandırma** veya filtreleme yapmalıyız.
- Gerçek dünyadaki senaryoyu test etmek istiyorsak, kullanıcıların kendi rastgele tıklayan kullanıcılar gürültü
  oluşturur, ama doğru değerlendiren küçük bir kesim bile sinyal üretmek için yeterli olabilir.

----------  

### **From Comparative Performance to Absolute Performance**

(“Göreceli iyi” ile “yeterince iyi” arasındaki fark)

- Karşılaştırmalı değerlendirme “hangi model daha iyi” sorusunu cevaplar, **“bu model yeterince iyi mi?”** sorusunu
  cevaplamaz.
- Model B, A’dan %51 oranla daha çok tercih ediliyor olabilir. Ama bu:
    - B çok iyi, A çok kötü olabilir
    - İkisi de kötü olabilir
    - İkisi de çok iyi olabilir
- Ayrıca bu farkın **gerçek iş çıktısına etkisi uygulamaya göre değişir.**
- Model B daha pahalıysa, %1’lik tercih farkı değişimi haklı çıkarmayabilir.

### ** Dikkat edilmesi gerekenler:**

- Kendi projemiz varsa: yalnızca hangi model daha çok tercih ediliyor değil, aynı zamanda **maliyet-fayda analizi** de
  yapmamız gerek.
- Müşteri destek gibi uygulamalarda başarı oranı (% kaç ticket çözüldü) gibi **mutlak performans metriği** de gerekir.

## The Future of Comparative Evaluation

### **Neden Geleceği Var?**

Bu kısımda yazar şunu söylüyor:

> “Bunca sınırlamaya rağmen karşılaştırmalı değerlendirmenin (comparative evaluation) hâlâ çok güçlü bir geleceği
> olabilir.”

###   * Neden?**

- **İnsanlar puanlamaya göre karşılaştırma yapmayı daha kolay buluyor.**
- Özellikle LLM’ler çok iyi hale geldikçe, bir cevaba 9 mu 10 mu verileceğini söylemek zor.
- Ancak iki cevaptan hangisi daha iyi dendiğinde bunu seçmek daha kolay.

**Dikkat:** Bu, model performansı “human level” veya “beyond human” hale geldiğinde puanlamanın anlamsızlaşacağına dair
çok önemli bir argümandır. Bu yüzden comparative evaluation **“insan farkını ölçmeye devam edebilir”**.
  
----------  

### **Benchmarklara Göre Daha Dayanıklı**

- **Benchmark’lar** sabit veri kümeleri ile sınırlı, bir model çok iyi hale geldiğinde benchmark’ı “bitirebilir” (
  örneğin: %100 skor).
- Ama karşılaştırmalı değerlendirme **sürekli gelişmeye açık** çünkü daha güçlü modeller çıktıkça yeni kıyaslar ortaya
  çıkar.

Bu durum, karşılaştırmalı değerlendirmenin **asla “tamamlandı” denemeyecek** bir yöntem olduğunu gösteriyor.
  
----------  

### **Manipülasyona Karşı Daha Güvenli**

- Benchmark’larda “data leakage” riski var (model test verisine aşina olabilir).
- Ama karşılaştırmalı değerlendirme için **“önceden ezberlenmiş doğru cevap” gibi bir şey yok**.
    - O yüzden **bazı halk sıralamaları bu yüzden daha güvenilir görülüyor**.

**Dikkat:** LMSYS, Chatbot Arena gibi sistemler bu mantıkla kuruldu. Modelin “kim olduğu” gizleniyor, karşılaştırma
tamamlandıktan sonra açıklanıyor.
  
----------  

### **Ekstra Bilgi Sağlar**

- Özellikle **offline değerlendirmelerde** klasik benchmarklara ek olarak kullanılabilir.
- **Online testlerde (A/B test)** ile birlikte kullanılabilir.
    - Örnek: Kullanıcıya iki farklı yanıt gösterilip hangisini seçtiği izlenebilir.

----------  

# Summary

Bu kısım tüm bölümün ve önceki alt başlıkların genel özetidir. Şu ana kadar öğrendiğimiz kavramları kapsar.

## **Güçlü modeller = Güçlü riskler**

- Modeller geliştikçe hata yapma potansiyeli de artıyor (örneğin, hallucination, toksik çıktı).
- Bu yüzden **değerlendirme daha da kritik** hale geliyor.

----------  

## **Otomatik Değerlendirmeye Odak**

- Bu bölümde özellikle insan değil, **otomatik değerlendirme yöntemlerine** odaklanıldı.
    - Cross-entropy, perplexity gibi metrikler (daha çok dil modeli bazlı)
    - Exact match / lexical / semantic benzerlik ölçümleri
    - AI as a judge (AI’nin hakem olarak kullanılması)

----------  

## **Subjektif Skorlar Güvenilmezdir**

- AI judges farklı sonuçlar verebilir.
- Skorların birbiriyle **karşılaştırılması zordur**.
- AI judge’lar değiştiği için **zamanla tutarsız hale gelir**.

**Dikkat:** Eğer ürünün bir sürümünü AI judge ile test ediyorsak, versiyon değişince aynı metrik aynı sonucu
vermeyebilir. **Regresyon testi yapılamaz.**
  
----------  

## **Preference Signal = Karşılaştırmanın Temeli**

- Post-training alignment (ör. RLHF) ve comparative evaluation **user preference sinyali** ister.
- Bu sinyal **çok pahalı**.
    - Bu yüzden bu sinyali tahmin etmeye çalışan “preference model”’ler geliştiriliyor.

----------  

## **Karşılaştırmalı Değerlendirme Yeni Yeni Yaygınlaşıyor**

- Perplexity gibi metrikler eski, ama comparative evaluation ve AI judge **foundation model’larla** birlikte
  yaygınlaştı.
- Hâlâ birçok ekip bu yöntemleri nasıl süreçlerine katacaklarını anlamaya çalışıyor.