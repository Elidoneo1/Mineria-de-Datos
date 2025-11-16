import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos de NLTK (solo primera vez)
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
except:
    print("Los recursos de NLTK ya están descargados")

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Palabras específicas de basketball para excluir
        self.basketball_stopwords = {
            'nba', 'basketball', 'basket', 'ball', 'game', 'games', 'team', 'teams',
            'player', 'players', 'play', 'playing', 'played', 'season', 'seasons'
        }
        
        self.all_stopwords = self.stop_words.union(self.basketball_stopwords)
    
    def preprocess_text(self, text, use_lemmatization=True, remove_numbers=True):
        """
        Preprocesamiento completo de texto
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover números si se solicita
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remover puntuación y caracteres especiales
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Filtrar stopwords y tokens muy cortos
        tokens = [token for token in tokens if token not in self.all_stopwords and len(token) > 2]
        
        # Lematización o stemming
        if use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text):
        """
        Análisis de sentimiento usando TextBlob
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def get_top_ngrams(self, text, n=1, top_k=20):
        """
        Obtener n-gramas más frecuentes
        """
        tokens = text.split()
        if n == 1:
            ngrams = tokens
        else:
            ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        counter = Counter(ngrams)
        return counter.most_common(top_k)
    
    def create_advanced_wordcloud(self, text, title, filename, 
                                width=800, height=400, 
                                background_color='white',
                                colormap='viridis'):
        """
        Crear wordcloud avanzado con múltiples opciones
        """
        # Crear máscara opcional (podrías añadir una forma específica)
        # Por ahora usamos None para forma rectangular
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            stopwords=self.all_stopwords,
            max_words=200,
            min_font_size=8,
            max_font_size=100,
            random_state=42,
            relative_scaling=0.5,
            collocations=True
        ).generate(text)
        
        # Crear visualización
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud: {title}', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'Practica9/{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return wordcloud

# ANÁLISIS 1: NOMBRES DE JUGADORES
print("=" * 80)
print("ANÁLISIS 1: NOMBRES DE JUGADORES NBA")
print("=" * 80)

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Total de jugadores únicos: {df['name'].nunique()}")

# Inicializar analizador
analyzer = TextAnalyzer()

# Preprocesar nombres de jugadores
all_names = ' '.join(df['name'].astype(str))
preprocessed_names = analyzer.preprocess_text(all_names, use_lemmatization=False)

print(f"\nESTADÍSTICAS DE NOMBRES:")
print(f"Total de caracteres: {len(all_names):,}")
print(f"Total de palabras (preprocesadas): {len(preprocessed_names.split()):,}")

# Análisis de apellidos más comunes
names_list = df['name'].str.split().explode()
surnames = names_list[names_list.str.len() > 2]  # Filtrar nombres muy cortos
surname_counts = surnames.value_counts().head(20)

print(f"\nAPELLIDOS MÁS COMUNES EN LA NBA:")
for surname, count in surname_counts.head(10).items():
    print(f"  {surname}: {count} jugadores")

# Wordcloud de nombres
print("\nGENERANDO WORDCLOUD DE NOMBRES...")
name_wordcloud = analyzer.create_advanced_wordcloud(
    preprocessed_names, 
    "Nombres de Jugadores NBA (2000-2009)",
    "wordcloud_names",
    colormap='plasma'
)

# ANÁLISIS 2: POSICIONES Y EQUIPOS
print("\n" + "=" * 80)
print("ANÁLISIS 2: POSICIONES Y EQUIPOS")
print("=" * 80)

# Combinar posiciones y equipos
positions_text = ' '.join(df['position'].astype(str) * 3)  # Peso extra para posiciones
teams_text = ' '.join(df['team'].astype(str))

basketball_terms = positions_text + " " + teams_text
preprocessed_basketball = analyzer.preprocess_text(basketball_terms)

print("TÉRMINOS DE BASKETBALL MÁS COMUNES:")
basketball_ngrams = analyzer.get_top_ngrams(preprocessed_basketball, n=1, top_k=15)
for term, count in basketball_ngrams:
    print(f"  {term}: {count}")

# Wordcloud de términos de basketball
print("\nGENERANDO WORDCLOUD DE TÉRMINOS NBA...")
basketball_wordcloud = analyzer.create_advanced_wordcloud(
    preprocessed_basketball,
    "Términos de Basketball (Posiciones y Equipos)",
    "wordcloud_basketball_terms",
    colormap='cool'
)

# ANÁLISIS 3: COMBINACIÓN DE TEXTO COMPLETO
print("\n" + "=" * 80)
print("ANÁLISIS 3: TEXTO COMPLETO DEL DATASET")
print("=" * 80)

# Combinar todas las columnas de texto
all_text_data = ""
text_columns = ['name', 'position', 'team']

for col in text_columns:
    if col in df.columns:
        column_text = ' '.join(df[col].astype(str))
        all_text_data += column_text + " "

preprocessed_all = analyzer.preprocess_text(all_text_data)

print(f"ESTADÍSTICAS DEL TEXTO COMPLETO:")
print(f"Palabras únicas: {len(set(preprocessed_all.split())):,}")
print(f"Palabra más larga: {max(preprocessed_all.split(), key=len)}")

# Análisis de n-gramas
print(f"\n🔤 N-GRAMAS MÁS FRECUENTES:")

for n in [1, 2, 3]:
    ngrams = analyzer.get_top_ngrams(preprocessed_all, n=n, top_k=8)
    print(f"\n{n}-gramas:")
    for ngram, count in ngrams:
        print(f"  '{ngram}': {count}")

# Wordcloud completo
print("\nGENERANDO WORDCLOUD COMPLETO...")
complete_wordcloud = analyzer.create_advanced_wordcloud(
    preprocessed_all,
    "Análisis Textual Completo - NBA Dataset",
    "wordcloud_complete",
    colormap='viridis'
)

# VISUALIZACIONES ADICIONALES
print("\n" + "=" * 80)
print("VISUALIZACIONES ADICIONALES")
print("=" * 80)

# 1. GRÁFICO DE BARRAS - Apellidos más comunes
plt.figure(figsize=(12, 8))
top_surnames = surname_counts.head(15)
plt.barh(top_surnames.index, top_surnames.values, color=sns.color_palette("husl", len(top_surnames)))
plt.xlabel('Frecuencia')
plt.ylabel('Apellido')
plt.title('Apellidos Más Comunes en la NBA (2000-2009)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Practica9/top_surnames.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. GRÁFICO DE BARRAS - Posiciones más comunes
plt.figure(figsize=(10, 6))
position_counts = df['position'].value_counts()
plt.bar(position_counts.index, position_counts.values, color='lightcoral')
plt.xlabel('Posición')
plt.ylabel('Número de Jugadores')
plt.title('Distribución de Posiciones en la NBA')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Practica9/position_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. GRÁFICO DE TORTA - Distribución de equipos (top 10)
plt.figure(figsize=(10, 8))
team_counts = df['team'].value_counts().head(10)
plt.pie(team_counts.values, labels=team_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Equipos con Más Jugadores (2000-2009)')
plt.tight_layout()
plt.savefig('Practica9/team_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ANÁLISIS DE LONGITUD DE NOMBRES
plt.figure(figsize=(12, 6))

# Longitud de nombres completos
name_lengths = df['name'].str.split().str.join('').str.len()

plt.subplot(1, 2, 1)
plt.hist(name_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Longitud del Nombre (caracteres)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Longitud de Nombres')
plt.grid(True, alpha=0.3)

# Número de palabras por nombre
word_counts = df['name'].str.split().str.len()

plt.subplot(1, 2, 2)
word_count_dist = word_counts.value_counts().sort_index()
plt.bar(word_count_dist.index, word_count_dist.values, color='lightgreen', alpha=0.7)
plt.xlabel('Número de Palabras en el Nombre')
plt.ylabel('Frecuencia')
plt.title('Número de Palabras por Nombre')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Practica9/name_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ANÁLISIS DE PATRONES ESPECÍFICOS
print("\n" + "=" * 80)
print("ANÁLISIS DE PATRONES ESPECÍFICOS")
print("=" * 80)

# Patrones en nombres
print("🔍 PATRONES EN NOMBRES DE JUGADORES:")

# Nombres que contienen "Jr", "Sr", "III", etc.
special_patterns = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
for pattern in special_patterns:
    count = df['name'].str.lower().str.contains(pattern, na=False).sum()
    if count > 0:
        print(f"  Nombres con '{pattern.upper()}': {count}")

# Nombres con apóstrofe
apostrophe_count = df['name'].str.contains("'", na=False).sum()
print(f"  Nombres con apóstrofe: {apostrophe_count}")

# Jugadores con el mismo apellido (posibles familiares)
surname_duplicates = surnames.value_counts()
common_surnames = surname_duplicates[surname_duplicates > 1]
print(f"\n👥 Apellidos compartidos por múltiples jugadores: {len(common_surnames)}")

# Mostrar algunos ejemplos
print("\nEjemplos de apellidos compartidos:")
for surname, count in common_surnames.head(5).items():
    players = df[df['name'].str.contains(surname, case=False, na=False)]['name'].tolist()
    print(f"  {surname} ({count}): {', '.join(players[:3])}")

# COMPARACIÓN ENTRE WORDCLOUDS
print("\n" + "=" * 80)
print("COMPARACIÓN ENTRE WORDCLOUDS")
print("=" * 80)

# Crear figura comparativa
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Wordcloud 1: Nombres
axes[0, 0].imshow(name_wordcloud, interpolation='bilinear')
axes[0, 0].set_title('Nombres de Jugadores', fontsize=14)
axes[0, 0].axis('off')

# Wordcloud 2: Términos basketball
axes[0, 1].imshow(basketball_wordcloud, interpolation='bilinear')
axes[0, 1].set_title('Términos de Basketball', fontsize=14)
axes[0, 1].axis('off')

# Wordcloud 3: Completo
axes[1, 0].imshow(complete_wordcloud, interpolation='bilinear')
axes[1, 0].set_title('Análisis Completo', fontsize=14)
axes[1, 0].axis('off')

# Gráfico de frecuencias
top_words = analyzer.get_top_ngrams(preprocessed_all, n=1, top_k=10)
words, counts = zip(*top_words)
axes[1, 1].barh(words, counts, color='lightblue')
axes[1, 1].set_title('Top 10 Palabras Más Frecuentes', fontsize=14)
axes[1, 1].set_xlabel('Frecuencia')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('Practica9/wordclouds_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# EXPORTACIÓN DE RESULTADOS
print("\n" + "=" * 80)
print("EXPORTACIÓN DE RESULTADOS")
print("=" * 80)

# Crear resumen de análisis
analysis_summary = {
    'total_jugadores': len(df),
    'jugadores_unicos': df['name'].nunique(),
    'total_palabras_preprocesadas': len(preprocessed_all.split()),
    'palabras_unicas': len(set(preprocessed_all.split())),
    'apellidos_comunes': len(common_surnames),
    'posiciones_unicas': df['position'].nunique(),
    'equipos_unicos': df['team'].nunique()
}

summary_df = pd.DataFrame([analysis_summary])
summary_df.to_csv('Practica9/analysis_summary.csv', index=False)

# Exportar frecuencias de palabras
word_frequencies = analyzer.get_top_ngrams(preprocessed_all, n=1, top_k=50)
freq_df = pd.DataFrame(word_frequencies, columns=['word', 'frequency'])
freq_df.to_csv('Practica9/word_frequencies.csv', index=False)

print("📁 ARCHIVOS EXPORTADOS:")
print("  Practica9/wordcloud_names.png")
print("  Practica9/wordcloud_basketball_terms.png")
print("  Practica9/wordcloud_complete.png")
print("  Practica9/wordclouds_comparison.png")
print("  Practica9/top_surnames.png")
print("  Practica9/position_distribution.png")
print("  Practica9/team_distribution.png")
print("  Practica9/name_analysis.png")
print("  Practica9/analysis_summary.csv")
print("  Practica9/word_frequencies.csv")

print(f"\nHALLAZGOS PRINCIPALES:")
print(f"  • {analysis_summary['jugadores_unicos']} jugadores únicos analizados")
print(f"  • {analysis_summary['palabras_unicas']} palabras únicas identificadas")
print(f"  • {analysis_summary['apellidos_comunes']} apellidos compartidos entre jugadores")
print(f"  • Patrones identificados en nombres y terminología basketball")

print("\n" + "=" * 80)
print("¡ANÁLISIS TEXTUAL Y WORD CLOUDS COMPLETADO!")
print("=" * 80)