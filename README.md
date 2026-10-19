# Adversarial Robustness: CNN vs Vision Transformer
## Amaç

CNN (ResNet-18) ve Vision Transformer (ViT) modellerinin adversarial saldırılara
karşı dayanıklılığını karşılaştırmak ve adversarial training'in bu farkı nasıl
değiştirdiğini ölçmek.

**Araştırma sorusu:** Aynı saldırılar ve aynı ε altında CNN ve ViT ne kadar dayanıklı?
Adversarial training sonrası hangisi daha dayanıklı oluyor ve bunun normal
doğruluğa maliyeti ne?

## Yöntem

- **Veri seti:** CIFAR-10
- **Modeller:** ResNet-18, ViT
- **Saldırılar:** FGSM, PGD (ana ölçüm), AutoAttack (doğrulama)
- **Tehdit modeli:** White-box, L∞, ε = 8/255
- **Savunma:** Adversarial training (PGD tabanlı)
- **Ölçütler:** Clean accuracy, robust accuracy, ε eğrisi

## İlk Sonuçlar (Ön Deney)

Basit bir CNN, CIFAR-10 üzerinde 8 epoch eğitildi.
Notebook: [`01_fgsm_pgd_cnn.ipynb`](01_fgsm_pgd_cnn.ipynb)

| ε | FGSM | PGD (10 adım) |
|---|---|---|
| 0 (saldırı yok) | %72.4 | %72.4 |
| 1/255 | %54.3 | %54.2 |
| 2/255 | %38.0 | %34.5 |
| 4/255 | %16.9 | %10.1 |
| 8/255 | %2.9 | %0.4 |

**Gözlemler:**
- Gözle neredeyse fark edilmeyen gürültü doğruluğu %72'den %0.4'e düşürüyor.
- PGD her ε değerinde FGSM'ye eşit veya daha güçlü; FGSM dayanıklılığı olduğundan yüksek gösteriyor.
- Bu yüzden ana ölçüm olarak PGD, son doğrulama için AutoAttack kullanılacak.

## Araçlar

PyTorch · torchvision · Google Colab (T4 GPU)
