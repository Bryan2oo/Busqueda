import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# =========================================================
# CONFIGURACIÓN - RUTA ABSOLUTA
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DOCUMENTOS = os.path.join(SCRIPT_DIR, "documentos")

print(f"📁 Ruta del script: {SCRIPT_DIR}")
print(f"📂 Buscando documentos en: {CARPETA_DOCUMENTOS}")

# =========================================================
# STOP WORDS EN ESPAÑOL (para AMBOS vectorizadores)
# =========================================================
STOP_WORDS_ES = [
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
    'de', 'del', 'al', 'en', 'con', 'por', 'para', 'sin', 'sobre',
    'y', 'o', 'pero', 'si', 'no', 'que', 'como', 'cuando', 'donde',
    'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas',
    'mi', 'tu', 'su', 'nuestro', 'vuestro',
    'me', 'te', 'se', 'nos', 'os',
    'es', 'son', 'era', 'eran', 'fue', 'fueron', 'ser', 'estar',
    'ha', 'han', 'he', 'hemos', 'haber',
    'lo', 'le', 'les', 'la', 'las', 'los'
]

# =========================================================
# 1. CARGAR DOCUMENTOS
# =========================================================
def cargar_documentos(ruta_carpeta):
    documentos = []
    nombres_archivos = []
    ruta = Path(ruta_carpeta)

    if not ruta.exists():
        ruta.mkdir(parents=True, exist_ok=True)
        return documentos, nombres_archivos

    for archivo in sorted(ruta.glob("*.txt")):
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read().strip()
                if contenido:
                    documentos.append(contenido)
                    nombres_archivos.append(archivo.name)
        except Exception as e:
            print(f"❌ Error: {e}")

    return documentos, nombres_archivos


# =========================================================
# 2. CLASE QUE MUESTRA BoW y TF-IDF
# =========================================================
class ComparadorBoW_TFIDF:
    def __init__(self, ruta_carpeta):
        self.ruta_carpeta = ruta_carpeta
        self.documentos = []
        self.nombres_archivos = []
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.matriz_bow = None
        self.matriz_tfidf = None

    def indexar(self):
        """Indexa documentos con AMBOS métodos"""
        print("\n📂 Cargando documentos...")
        self.documentos, self.nombres_archivos = cargar_documentos(self.ruta_carpeta)

        if len(self.documentos) == 0:
            print(f"⚠️ Agrega archivos .txt a '{self.ruta_carpeta}'")
            return False

        print(f"✅ {len(self.documentos)} documentos encontrados")

        # ========== BAG OF WORDS ==========
        print("\n⚙️ Creando matriz BoW (CountVectorizer)...")
        self.bow_vectorizer = CountVectorizer(
            stop_words=STOP_WORDS_ES,  # ✅ Lista personalizada
            ngram_range=(1, 1)
        )
        self.matriz_bow = self.bow_vectorizer.fit_transform(self.documentos)

        # ========== TF-IDF ==========
        print("⚙️ Creando matriz TF-IDF (TfidfVectorizer)...")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS_ES,  # ✅ CORRECCIÓN: Usar lista en vez de 'spanish'
            max_features=10000,
            ngram_range=(1, 1)
        )
        self.matriz_tfidf = self.tfidf_vectorizer.fit_transform(self.documentos)

        print("✅ Indexación completa")
        return True

    def mostrar_diferencias(self):
        """Muestra las diferencias entre BoW y TF-IDF"""
        print("\n" + "=" * 80)
        print("📊 COMPARATIVA: BAG OF WORDS vs TF-IDF")
        print("=" * 80)

        vocab_bow = self.bow_vectorizer.get_feature_names_out()
        vocab_tfidf = self.tfidf_vectorizer.get_feature_names_out()

        print(f"\n📏 Dimensiones:")
        print(f"   BoW:    {self.matriz_bow.shape}")
        print(f"   TF-IDF: {self.matriz_tfidf.shape}")

        print(f"\n🔢 MATRIZ BoW (primeros 5 términos):")
        print(f"   {vocab_bow[:5]}")
        print(self.matriz_bow.toarray()[:, :5])

        print(f"\n🔢 MATRIZ TF-IDF (primeros 5 términos):")
        print(f"   {vocab_tfidf[:5]}")
        print(np.round(self.matriz_tfidf.toarray()[:, :5], 4))

        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   BoW - Máximo: {self.matriz_bow.max():.0f}, Mínimo: {self.matriz_bow.min():.0f}")
        print(f"   TF-IDF - Máximo: {self.matriz_tfidf.max():.4f}, Mínimo: {self.matriz_tfidf.min():.4f}")

        print("=" * 80)

    def buscar_comparativo(self, texto_consulta, top_n=3):
        """Busca con AMBOS métodos y muestra resultados comparativos"""
        print("\n" + "=" * 80)
        print(f"🔍 CONSULTA: '{texto_consulta}'")
        print("=" * 80)

        vector_bow = self.bow_vectorizer.transform([texto_consulta])
        vector_tfidf = self.tfidf_vectorizer.transform([texto_consulta])

        sim_bow = cosine_similarity(vector_bow, self.matriz_bow)[0]
        sim_tfidf = cosine_similarity(vector_tfidf, self.matriz_tfidf)[0]

        idx_bow = np.argsort(sim_bow)[::-1]
        idx_tfidf = np.argsort(sim_tfidf)[::-1]

        print("\n📄 MÉTODO 1: BAG OF WORDS (BoW)")
        print("-" * 80)
        print(f"{'POS':<5} | {'SIMILITUD':<12} | {'ARCHIVO':<25}")
        print("-" * 80)

        for i in range(min(top_n, len(idx_bow))):
            idx = idx_bow[i]
            if sim_bow[idx] > 0:
                print(f"{i + 1:<5} | {sim_bow[idx]:.4f}       | {self.nombres_archivos[idx]}")

        print("\n📄 MÉTODO 2: TF-IDF")
        print("-" * 80)
        print(f"{'POS':<5} | {'SIMILITUD':<12} | {'ARCHIVO':<25}")
        print("-" * 80)

        for i in range(min(top_n, len(idx_tfidf))):
            idx = idx_tfidf[i]
            if sim_tfidf[idx] > 0:
                print(f"{i + 1:<5} | {sim_tfidf[idx]:.4f}       | {self.nombres_archivos[idx]}")

        print("=" * 80)

        print("\n💡 ANÁLISIS:")
        mejor_bow = self.nombres_archivos[idx_bow[0]] if sim_bow[idx_bow[0]] > 0 else "Ninguno"
        mejor_tfidf = self.nombres_archivos[idx_tfidf[0]] if sim_tfidf[idx_tfidf[0]] > 0 else "Ninguno"

        if mejor_bow == mejor_tfidf:
            print(f"   ✅ Ambos métodos coinciden: '{mejor_bow}'")
        else:
            print(f"   ⚠️ Resultados diferentes:")
            print(f"      BoW:    '{mejor_bow}'")
            print(f"      TF-IDF: '{mejor_tfidf}'")

        print("=" * 80)


# =========================================================
# 3. MENÚ PRINCIPAL
# =========================================================
def main():
    print("\n" + "=" * 80)
    print("📊 GRUPO 3: REPRESENTACIÓN ESTADÍSTICA")
    print("   Comparativa: Bag of Words (BoW) vs TF-IDF")
    print("=" * 80)

    comparador = ComparadorBoW_TFIDF(CARPETA_DOCUMENTOS)

    if not comparador.indexar():
        return

    comparador.mostrar_diferencias()

    print("\n" + "=" * 80)
    print("📝 MODO BÚSQUEDA (escribe 'salir' para terminar)")
    print("=" * 80)

    while True:
        try:
            consulta = input("\n🔍 Consulta: ").strip()

            if consulta.lower() in ['salir', 'exit', 'q']:
                print("\n👋 ¡Fin del programa!")
                break

            if not consulta:
                continue

            comparador.buscar_comparativo(consulta, top_n=5)

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()