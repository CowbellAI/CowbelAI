import json

# JSON dosyasını okuyarak sözlüğü yükleme fonksiyonu
def load_kayseri_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# Şive dönüştürme fonksiyonu
def convert_shive_to_standard(text, word_dict):
    for key, value in word_dict.items():
        if key in text:
            text = text.replace(key, value)
    return text

# Tam uygulama
def main():
    # JSON dosyasından kelimeleri yükle
    kayseri_to_standard = load_kayseri_words('Lehçe.json')
    
    while True:
        # Kullanıcıdan şive ile metin al
        kayseri_text = input("Kayseri şivesi ile bir şeyler yazın: ")
        
        # Çıkış komutu kontrolü
        if kayseri_text.lower() == 'çıkış':
            print("Uygulamadan çıkılıyor...")
            break
        
        # Şiveyi standart Türkçeye çevir
        standard_text = convert_shive_to_standard(kayseri_text, kayseri_to_standard)
        print(f"Standart Türkçe: {standard_text}")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()
