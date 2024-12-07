import tkinter as tk
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def kategori_filtrele():
    # movie.csv dosyasını oku
    filmler = pd.read_csv('movie.csv')

    # 'genres' sütunundaki kategorileri ayır ve say
    genres_expanded = filmler['genres'].str.get_dummies(sep='|')

    # Kategorileri say
    genre_counts = genres_expanded.sum().sort_values(ascending=False)

    # En sık bulunan 10 kategoriyi al
    top_10_genres = genre_counts.nlargest(10).index.tolist()

    # Top 10 kategori dışındaki filmleri sil
    # Kategorileri filtrele
    filmler['genres'] = filmler['genres'].apply(lambda x: '|'.join([genre for genre in x.split('|') if genre in top_10_genres]))

    # Kategorisi boş olan satırları sil
    filtered_movies = filmler[filmler['genres'].str.strip() != '']

    # Sonuçları yeni bir dosyaya yaz
    filtered_movies.to_csv('filtered_movies.csv', index=False)


def kullanici_filtrele():
    # rating.csv dosyasını oku
    ratings = pd.read_csv('rating.csv', usecols=['userId', 'movieId', 'rating'])
    ratings = ratings[ratings['rating'] >= 3.5]
    # Her kullanıcının izlediği film sayısını hesapla
    user_movie_counts = ratings['userId'].value_counts().reset_index()
    user_movie_counts.columns = ['userId', 'movie_count']

    # 800'den az film izleyen kullanıcıları filtrele
    filtered_users = user_movie_counts[user_movie_counts['movie_count'] >= 800]

    # İlgili kullanıcıları orijinal rating verisinden filtrele
    filtered_ratings = ratings[ratings['userId'].isin(filtered_users['userId'])]

    # Sonuçları yeni bir CSV dosyasına yaz
    filtered_ratings.to_csv('800den_fazla_film_izleyen_kullanicilar.csv', index=False)


def film_filtrele():
    # 800den_fazla_film_izleyen_kullanicilar.csv dosyasını oku
    filtrelenmis_rating = pd.read_csv('800den_fazla_film_izleyen_kullanicilar.csv', usecols=['userId', 'movieId', 'rating'])

    # İzlenen film sayısını hesapla
    izlenen_film_sayisi = filtrelenmis_rating['movieId'].value_counts().reset_index()
    izlenen_film_sayisi.columns = ['movieId', 'movie_count']

    # 200'den az izlenen filmleri filtrele
    fazla_izlenen_filmler = izlenen_film_sayisi[izlenen_film_sayisi['movie_count'] >= 200]

    # İlgili kullanıcıları orijinal rating verisinden filtrele
    filtrelenmis_filmler = filtrelenmis_rating[filtrelenmis_rating['movieId'].isin(fazla_izlenen_filmler['movieId'])]

    # Sonuçları yeni bir CSV dosyasına yaz
    filtrelenmis_filmler.to_csv('200den_fazla_izlenen_filmler.csv', index=False)

def ortak_filmler():
    # CSV dosyalarını oku
    filtered_movies = pd.read_csv('filtered_movies.csv')
    rated_movies = pd.read_csv('200den_fazla_izlenen_filmler.csv')

    # Her iki dosyadaki film ID'lerini karşılaştır
    common_movies = set(filtered_movies['movieId']).intersection(set(rated_movies['movieId']))

    # Ortak film ID'leri ile filtrele
    filtered_movies = filtered_movies[filtered_movies['movieId'].isin(common_movies)]
    rated_movies = rated_movies[rated_movies['movieId'].isin(common_movies)]

    # Güncellenmiş dosyaları yaz
    filtered_movies.to_csv('filtered_movies.csv', index=False)
    rated_movies.to_csv('200den_fazla_izlenen_filmler.csv', index=False)

def kullanici_film_listesi():
    # 200den_fazla_izlenen_filmler.csv dosyasını oku
    ratings = pd.read_csv('200den_fazla_izlenen_filmler.csv')

    # Her kullanıcının izlediği film listesini oluştur
    user_movie_list = ratings.groupby('userId')['movieId'].apply(lambda x: ', '.join(map(str, x))).reset_index()

    # Sütun isimlerini düzenle
    user_movie_list.columns = ['userId', 'watched_movies']

    # Sonuçları yeni bir CSV dosyasına yaz
    user_movie_list.to_csv('user_watched_movies.csv', index=False)

def matrix_olusutr():
    # kategories_movie.csv ve filtrelenmis_izleyici_film_tablosu.csv dosyalarını oku
    movies_df = pd.read_csv('filtered_movies.csv')
    user_movies_df = pd.read_csv('user_watched_movies.csv')

    # 'İzlenen Filmler' sütununu liste olarak yorumla
    user_movies_df['watched_movies'] = user_movies_df['watched_movies'].apply(eval)

    # Tüm izleyici ve film eşleşmelerini içeren bir DataFrame oluştur
    user_movie_data = []
    for _, row in user_movies_df.iterrows():
        user_id = row['userId']
        watched_movies = row['watched_movies']
        for movie_id in watched_movies:
            if movie_id in movies_df['movieId'].values:  # Filmin kategories_movie.csv'de olup olmadığını kontrol et
                user_movie_data.append((user_id, movie_id))

    # Kullanıcı ve film ID'lerine göre bir DataFrame oluştur
    user_movie_df = pd.DataFrame(user_movie_data, columns=['userId', 'movieId'])

    # Kullanıcı-film matrisini pivot tablosu olarak oluştur
    user_movie_matrix = user_movie_df.pivot_table(index='userId', columns='movieId', aggfunc='size', fill_value=0)

    # Sütun isimlerini film ID'lerine göre düzenleyin
    user_movie_matrix.index.name = 'userId'
    user_movie_matrix.columns = [f'{int(col)}' for col in user_movie_matrix.columns]

    # Sonucu yeni bir CSV dosyasına kaydedin
    user_movie_matrix.to_csv('matrix.csv')

# kategori_filtrele()
# kullanici_filtrele()
# film_filtrele()
# ortak_filmler()
# kullanici_film_listesi()
# matrix_olusutr()
###################################################################


# Verilerinizi yükleyin
user_movie_matrix = pd.read_csv('matrix.csv', index_col=0)
min_support = 0.3

frequent_itemsets = apriori(user_movie_matrix, min_support=min_support, use_colnames=True, max_len=2, low_memory=True,)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
print(rules)
# Ağaç düğüm yapısı oluşturma
class RuleNode:
    def __init__(self):
        self.children = {}
        self.rule = None

class RuleTree:
    def __init__(self):
        self.root = RuleNode()
    
    def add_rule(self, antecedents, rule):
        node = self.root
        # Antecedents setini sıralı bir liste olarak işleyin
        for item in sorted(antecedents):
            if item not in node.children:
                node.children[item] = RuleNode()
            node = node.children[item]
        node.rule = rule
    
    def find_rule(self, antecedents):
        node = self.root
        for item in sorted(antecedents):
            if item in node.children:
                node = node.children[item]
            else:
                return None
        return node.rule
    
    def print_tree(self, node=None, prefix=""):
        """ Ağacın tamamını yazdırmak için rekürsif fonksiyon """
        if node is None:
            node = self.root

        # Düğümdeki kuralı yazdır
        if node.rule is not None:
            print(f"{prefix} Rule: {node.rule}")

        # Çocukları yazdır
        for key, child in node.children.items():
            self.print_tree(child, prefix + f" {key} ->")

# Kuralları ağaca ekleme
rule_tree = RuleTree()
for _, rule in rules.iterrows():
    antecedents = rule['antecedents']
    rule_tree.add_rule(antecedents, rule)



user_watched_movies = pd.read_csv('user_watched_movies.csv')
filtrelenmis_movie = pd.read_csv('filtered_movies.csv')


class FilmOneriSistemiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Film Öneri Sistemi")
        
        # Pencere boyutunu ayarla
        self.root.geometry("500x600")
        
        # Öneri türü ve şartı saklamak için değişkenler
        self.oneri_turu = None
        self.sart = None
        self.kullanici = None

        # Ekranları oluştur
        self.create_frame1()
        self.create_frame2()
        self.create_frame3_personalized()
        self.create_frame4()

        # İlk ekranı göster
        self.show_frame(self.frame1)
        
    def show_frame(self, frame):
        """Verilen frame'i göster, diğerlerini gizle"""
        for f in [self.frame1, self.frame2, self.frame3_personalized, self.frame4]:
            f.place_forget()
        frame.place(relx=0.5, rely=0.5, anchor="center")

    def create_frame1(self):
        # İlk ekran - Öneri türü seçimi
        self.frame1 = tk.Frame(self.root)
        
        label = tk.Label(self.frame1, text="Öneri Türü Seçin:")
        label.pack(pady=10)
        
        popular_button = tk.Button(self.frame1, text="Popüler Film Önerileri", command=lambda: self.select_oneri_turu("Popüler"))
        popular_button.pack(pady=5)
        
        personalized_button = tk.Button(self.frame1, text="Kişiselleştirilmiş Film Önerileri", command=lambda: self.select_oneri_turu("Kişiselleştirilmiş"))
        personalized_button.pack(pady=5)

    def create_frame2(self):
        # İkinci ekran - Öneri şartı seçimi
        self.frame2 = tk.Frame(self.root)
        
        label2 = tk.Label(self.frame2, text="Öneri Şartı Seçin:")
        label2.pack(pady=10)
        
        movie_button = tk.Button(self.frame2, text="Filme Göre Öneri", command=lambda: self.select_sart("Film"))
        movie_button.pack(pady=5)
        
        genre_button = tk.Button(self.frame2, text="Türe Göre Öneri", command=lambda: self.select_sart("Tür"))
        genre_button.pack(pady=5)

        back_button = tk.Button(self.frame2, text="Geri", command=lambda: self.show_frame(self.frame1))
        back_button.pack(pady=5)

    def create_frame3_personalized(self):
        # Üçüncü ekran (Kişiselleştirilmiş) - Kullanıcı seçimi
        self.frame3_personalized = tk.Frame(self.root)
        
        label3 = tk.Label(self.frame3_personalized, text="Kullanıcı Seçin:")
        label3.pack(pady=10)
        
        # Kullanıcı listesi
        self.user_listbox = tk.Listbox(self.frame3_personalized)
        kullanicilar=user_watched_movies['userId'].unique()
        for user in kullanicilar:
            self.user_listbox.insert(tk.END, user)
        self.user_listbox.pack(pady=10)
        
        next_button = tk.Button(self.frame3_personalized, text="İlerle", command=self.select_user_and_proceed)
        next_button.pack(pady=5)
        back_button = tk.Button(self.frame3_personalized, text="Geri", command=lambda: self.show_frame(self.frame2))
        back_button.pack(pady=5)

    def create_frame4(self):
        # Dördüncü ekran - Film veya tür seçimi
        self.frame4 = tk.Frame(self.root)
        
        self.label4 = tk.Label(self.frame4, text="Seçim Yapın:")
        self.label4.pack(pady=10)
        
        self.selection_listbox = tk.Listbox(self.frame4)
        self.selection_listbox.pack(pady=10)
        
        # Öneri butonu
        finish_button = tk.Button(self.frame4, text="Öneriyi Göster", command=self.show_recommendation)
        finish_button.pack(pady=5)
        
        self.result_label = tk.Label(self.frame4, text="")
        self.result_label.pack(pady=10)
        if self.oneri_turu == "Kişiselleştirilmiş":
            back_button = tk.Button(self.frame4, text="Geri", command=lambda: self.show_frame(self.frame3_personalized))
            back_button.pack(pady=5)
        else:
            back_button = tk.Button(self.frame4, text="Geri", command=lambda: self.show_frame(self.frame2))
            back_button.pack(pady=5)

    def select_oneri_turu(self, tur):
        """Öneri türü seçildikten sonra ikinci ekrana geçiş yapar"""
        self.oneri_turu = tur
        self.show_frame(self.frame2)

    def select_sart(self, sart):
        """Öneri şartı seçildikten sonra üçüncü ekrana geçiş yapar"""
        self.sart = sart
        if self.oneri_turu == "Kişiselleştirilmiş":
            self.show_frame(self.frame3_personalized)
        else:
            # Popüler seçilmişse direkt dördüncü ekrana geçiş yapar
            if self.sart == "Film":
                self.show_frame4_movie_selection()
            elif self.sart == "Tür":
                self.show_frame4_genre_selection()

    def select_user_and_proceed(self):
        """Kullanıcıyı seçip dördüncü ekrana geçiş yapar"""
        selected_user = self.user_listbox.get(tk.ACTIVE)
        self.kullanici = selected_user
        if self.sart == "Film":
            self.show_frame4_movie_selection()
        elif self.sart == "Tür":
            self.show_frame4_genre_selection()

    def show_frame4_movie_selection(self):
        """Film seçimi ekranını gösterir"""
        self.selection_listbox.delete(0, tk.END)
        
        if self.oneri_turu == "Kişiselleştirilmiş":
            selected_user = self.user_listbox.get(tk.ACTIVE)
            selected_user = int(selected_user)
            user_movies = user_watched_movies[user_watched_movies['userId'] == selected_user]['watched_movies'].values[0]
            movie_ids = user_movies.split(',')
            self.label4.config(text="Kişiselleştirilmiş Film Seçin:")
        else:  # Popüler seçildiğinde
            movie_ids = filtrelenmis_movie['movieId'].tolist()
            self.label4.config(text="Popüler Film Seçin:")

        # MovieId'leri title'a dönüştürerek listbox'a ekleme
        for movie_id in movie_ids:
            # `movies` DataFrame'inde movieId ile title buluyoruz
            movie_title = filtrelenmis_movie[filtrelenmis_movie['movieId'] == int(movie_id)]['title'].values[0]
            # Listbox'a "title (movieId)" formatında ekleme yapıyoruz
            self.selection_listbox.insert(tk.END, f"{movie_title} ({movie_id})")

        self.show_frame(self.frame4)


    def show_frame4_genre_selection(self):
        """Tür seçimi ekranını gösterir"""
        self.selection_listbox.delete(0, tk.END)
        self.label4.config(text="Tür Seçin:")
        genres_series = filtrelenmis_movie['genres'].str.cat(sep='|').split('|')
        unique_genres = set(genres_series)
        for genre in unique_genres:
            self.selection_listbox.insert(tk.END, genre)
        self.show_frame(self.frame4)

    def show_recommendation(self):

        """Öneriyi gösterir"""
        selected_item = self.selection_listbox.get(tk.ACTIVE)#film veya tür değerini alır
        result_text = f"Seçilen Öneri Türü: {self.oneri_turu}\nSeçilen Şart: {self.sart}\n"

        if self.oneri_turu == "Kişiselleştirilmiş":#kişiselleştirilmiş bilgisi

            result_text += f"Kullanıcı: {self.kullanici}\n"
            secilen_kullanici = self.user_listbox.get(tk.ACTIVE)#seçili kullanıcıyı al
            secilen_kullanici = int(secilen_kullanici)#inte çevir
            kullanici_film = user_watched_movies[user_watched_movies['userId'] == secilen_kullanici]['watched_movies'].values[0]#o kullanıcının film listesini al
            filmler = kullanici_film.split(',')#liste olarak tanımlayıp ata

            if self.sart=="Film":#kişiselleştirilmiş filme göre

                secilen_film_id = selected_item.split('(')[-1].replace(')', '').strip()#seçilen filmi sadece id için ayıklar
                secilen_kurallar = rules[rules['antecedents'].apply(lambda x: str(secilen_film_id) in x)]#o iddeki kuralları alır

                if not secilen_kurallar.empty:
                    confidence_siralanmis_kurallar = secilen_kurallar.sort_values(by='confidence', ascending=False)
                    for index, row in confidence_siralanmis_kurallar.iterrows():
                        desteklenen_film = next(iter(row['consequents']))
                        if desteklenen_film not in filmler:
                            film_ismi = filtrelenmis_movie[filtrelenmis_movie['movieId'] == int(desteklenen_film)]['title'].values[0]#desteklenen değerin ismini alır id sayesinde
                            result_text += f"Önerilen film: {film_ismi}\n"#film ismini
                            break

                else:
                    result_text += f"{selected_item} numaralı film için bir öneri bulunamadı.\n"

            else:#kişiselleştirilmiş türe göre

                x=0#ilerde break bozmak için tanımladım
                film=0

                for film_id in filmler:#Kullanıcının izlediği her filmi kontrol et

                    filtrelenmis_kurallar = rules[rules['antecedents'].apply(lambda x: str(film_id) in x)]# Antecedents içinde movie_id'nin olduğu kuralları al
                    filtrelenmis_kurallar = filtrelenmis_kurallar.sort_values(by='confidence', ascending=False)

                    if not filtrelenmis_kurallar.empty:# Eğer uygun kurallar varsa, önerilen filmleri kontrol et
                        
                        for _, kural in filtrelenmis_kurallar.iterrows():#o kurallara tek tek bak

                            desteklenen_film=kural['consequents']# Frozenset içindeki değeri al
                            desteklenen_film_id = int(next(iter(desteklenen_film)))#idye çevir
                            film_satiri = filtrelenmis_movie[filtrelenmis_movie['movieId'] == desteklenen_film_id]#o iddeki filmi satırını al
                            film_türleri = film_satiri['genres'].values[0]#türlerini al
                            ayrilmis_film_türleri = film_türleri.split('|')#aynı filmde birkaç farklı tür olabilir o türleri ayır

                            if selected_item in ayrilmis_film_türleri and str(desteklenen_film_id) not in filmler:#seçilen tür film türü listesinde varsa ve kullanıcı o filmi izlememişse al
                                
                                film = desteklenen_film_id#film idsini al
                                film_ismi = filtrelenmis_movie[filtrelenmis_movie['movieId'] == int(film)]['title'].values[0]#o filmidnin titleını al
                                x=1
                                break

                    if x==1:
                        break#genel döngüyü kır film bulundu

                if not film==0:
                    result_text += f"Önerilen film: {film_ismi}\n"
                else:
                    result_text += f"{selected_item} Türü için bir öneri bulunamadı.\n"
                            

        else:#popülere göre öneri

            if self.sart=="Film":#popülerde filme göre

                secilen_film_id = selected_item.split('(')[-1].replace(')', '').strip()#film idsini al
                filtrelenmis_kurallar = rules[rules['antecedents'].apply(lambda x: str(secilen_film_id) in x)]#o iddeki kuralları al

                if not filtrelenmis_kurallar.empty:#boş değilse
                    
                    en_iyi_kural = filtrelenmis_kurallar.loc[filtrelenmis_kurallar['confidence'].idxmax()]#confidence değeri en yüksek olanı al
                    sonuc_filmi = list(en_iyi_kural['consequents'])[0]  # Sonuç ürünü al
                    film_ismi = filtrelenmis_movie[filtrelenmis_movie['movieId'] == int(sonuc_filmi)]['title'].values[0]#titlea çevir idyi
                    result_text += f"Önerilen film: {film_ismi}\n"

                else:
                    result_text += f"{selected_item} numaralı film için bir öneri bulunamadı.\n"

            else:#popülerde türe göre

                filtrelenmis_filmler = filtrelenmis_movie[filtrelenmis_movie['genres'].str.contains(selected_item, case=False)]#seçilen türdeki filmleri al
                turdeki_filmler = filtrelenmis_filmler['movieId'].tolist()#ıd listesine çevir
                # En yüksek antecedent support değerini bulmak için değişkenler
                max_deger = 0
                onerilen_film_id = None
                for film_id in turdeki_filmler:

                    film_id_string = str(film_id)#stringe çevir idyi
                    filmin_kurali = rule_tree.find_rule(frozenset([film_id_string]))#ağaçta bu iddeki kuralı al

                    if filmin_kurali is not None:
                        desteklenen_deger = filmin_kurali['antecedent support']#öncül support değerini al
                        if desteklenen_deger > max_deger:#max supporttan yüksek mi diye kontrol et
                            max_deger = desteklenen_deger#max supportu güncelle
                            onerilen_film_id = filmin_kurali['antecedents']  # film idsini güncelle

                # Sonucu göster
                if onerilen_film_id:
                    onerilen_film_id = int(next(iter(onerilen_film_id)))#idye çevir
                    film_ismi = filtrelenmis_movie[filtrelenmis_movie['movieId'] == int(onerilen_film_id)]['title'].values[0]
                    result_text += f"{selected_item} türü için önerilen film: {film_ismi}\n"

                else:
                    result_text += f"{selected_item} türü için bir öneri bulunamadı.\n"

        result_text += f"Seçim: {selected_item}"
        self.result_label.config(text=result_text)

# Ana uygulama penceresini oluştur
root = tk.Tk()
app = FilmOneriSistemiGUI(root)
root.mainloop()
