import json
import re

# JSON dosyasını okuyarak sözlüğü yükleme fonksiyonu
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# Şive dönüştürme fonksiyonu
def convert_shive_to_standard(text, word_dict):
    # Anahtar kelimeleri ve değerleri küçük harfe çevirip, boşlukları temizle
    text_cleaned = text.lower().strip()
    for key, value in word_dict.items():
        key_cleaned = key.lower().strip()
        # Anahtar kelimenin metnin içinde olup olmadığını kontrol et
        if key_cleaned in text_cleaned:
            text_cleaned = re.sub(r'\b{}\b'.format(re.escape(key_cleaned)), value, text_cleaned, flags=re.IGNORECASE)
    return text_cleaned

# Cevap üretim fonksiyonu
def generate_response(standard_text, responses):
    # Küçük harfe çevir ve boşlukları temizle
    standard_text = standard_text.lower().strip()
    for key, response in responses.items():
        # Anahtar kelimeleri küçük harfe çevirip, boşlukları temizle
        key_cleaned = key.lower().strip()
        if key_cleaned in standard_text:
            return response
    return responses.get("default", "Bu konu hakkında ne söyleyeceğimi bilemiyorum.")

# Tam uygulama
def main():
    # JSON dosyasından kelimeleri ve yanıtları yükle
    kayseri_to_standard = load_json('Sorular.json')
    responses = load_json('Soru-Cevap.json')
    
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
    
        # Standart Türkçeye çevrilen metne uygun cevap üret
        response = generate_response(standard_text, responses)
        print(f"Cevap: {response}")

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()
