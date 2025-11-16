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

# Descargar recursos de NLTK con manejo de errores
def download_nltk_resources():
    resources = [
        'punkt',
        'stopwords', 
        'wordnet',
        'vader_lexicon',
        'punkt_tab'  # Recurso adicional necesario
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Recurso {resource} descargado correctamente")
        except Exception as e:
            print(f"⚠ Error descargando {resource}: {e}")

print("Descargando recursos de NLTK...")
download_nltk_resources()
print("Descarga de recursos completada\n")

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class TextAnalyzer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Error inicializando NLTK: {e}")
            # Fallback a stopwords básicos
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
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
        Preprocesamiento completo de texto con manejo de errores
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        try:
            # Convertir a minúsculas
            text = text.lower()
            
            # Remover números si se solicita
            if remove_numbers:
                text = re.sub(r'\d+', '', text)
            
            # Remover puntuación y caracteres especiales
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Tokenización con fallback simple
            try:
                tokens = word_tokenize(text)
            except:
                # Fallback: tokenización simple por espacios
                tokens = text.split()
            
            # Filtrar stopwords y tokens muy cortos
            tokens = [token for token in tokens if token not in self.all_stopwords and len(token) > 2]
            
            # Lematización o stemming
            if use_lemmatization:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            else:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return ""
    
    def analyze_sentiment(self, text):
        """
        Análisis de sentimiento usando TextBlob
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0.0, 0.0
    
    def get_top_ngrams(self, text, n=1, top_k=20):
        """
        Obtener n-gramas más frecuentes
        """
        if not text:
            return []
            
        tokens = text.split()
        if len(tokens) < n:
            return []
            
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
        if not text or len(text.strip()) < 10:
            print(f"Texto insuficiente para wordcloud: {title}")
            return None
            
        try:
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
            
            # Crear directorio si no existe
            import os
            os.makedirs('Practica9', exist_ok=True)
            
            # Crear visualización
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud: {title}', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(f'Practica9/{filename}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return wordcloud
            
        except Exception as e:
            print(f"Error creando wordcloud {title}: {e}")
            return None

# ANÁLISIS 1: NOMBRES DE JUGADORES
print("=" * 80)
print("ANÁLISIS 1: NOMBRES DE JUGADORES NBA")
print("=" * 80)

# Cargar datos
try:
    df = pd.read_csv("edited-salaries.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Total de jugadores únicos: {df['name'].nunique()}")
except Exception as e:
    print(f"Error cargando datos: {e}")
    exit()

# Inicializar analizador
analyzer = TextAnalyzer()

# Preprocesar nombres de jugadores
all_names = ' '.join(df['name'].astype(str))
preprocessed_names = analyzer.preprocess_text(all_names, use_lemmatization=False)

print(f"\nESTADÍSTICAS DE NOMBRES:")
print(f"Total de caracteres: {len(all_names):,}")
print(f"Total de palabras (preprocesadas): {len(preprocessed_names.split())}")

# Análisis de apellidos más comunes
try:
    names_list = df['name'].str.split().explode()
    surnames = names_list[names_list.str.len() > 2]  # Filtrar nombres muy cortos
    surname_counts = surnames.value_counts().head(20)

    print(f"\nAPELLIDOS MÁS COMUNES EN LA NBA:")
    for surname, count in surname_counts.head(10).items():
        print(f"  {surname}: {count} jugadores")
except Exception as e:
    print(f"Error analizando apellidos: {e}")
    surname_counts = pd.Series()

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
try:
    positions_text = ' '.join(df['position'].astype(str) * 3)  # Peso extra para posiciones
    teams_text = ' '.join(df['team'].astype(str))

    basketball_terms = positions_text + " " + teams_text
    preprocessed_basketball = analyzer.preprocess_text(basketball_terms)

    print("TÉRMINOS DE BASKETBALL MÁS COMUNES:")
    basketball_ngrams = analyzer.get_top_ngrams(preprocessed_basketball, n=1, top_k=15)
    for term, count in basketball_ngrams:
        print(f"  {term}: {count}")
except Exception as e:
    print(f"Error procesando términos basketball: {e}")
    preprocessed_basketball = ""

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
try:
    all_text_data = ""
    text_columns = ['name', 'position', 'team']

    for col in text_columns:
        if col in df.columns:
            column_text = ' '.join(df[col].astype(str))
            all_text_data += column_text + " "

    preprocessed_all = analyzer.preprocess_text(all_text_data)

    print(f"ESTADÍSTICAS DEL TEXTO COMPLETO:")
    print(f"Palabras únicas: {len(set(preprocessed_all.split()))}")
    if preprocessed_all.split():
        print(f"Palabra más larga: {max(preprocessed_all.split(), key=len)}")
    else:
        print("No hay palabras para analizar")
except Exception as e:
    print(f"Error procesando texto completo: {e}")
    preprocessed_all = ""

# Análisis de n-gramas
print(f"\n N-GRAMAS MÁS FRECUENTES:")

for n in [1, 2, 3]:
    ngrams = analyzer.get_top_ngrams(preprocessed_all, n=n, top_k=8)
    if ngrams:
        print(f"\n{n}-gramas:")
        for ngram, count in ngrams:
            print(f"  '{ngram}': {count}")
    else:
        print(f"\nNo hay {n}-gramas disponibles")

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

# Crear directorio para guardar imágenes
import os
os.makedirs('Practica9', exist_ok=True)

# 1. GRÁFICO DE BARRAS - Apellidos más comunes
try:
    plt.figure(figsize=(12, 8))
    if len(surname_counts) > 0:
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
    else:
        print("No hay datos suficientes para el gráfico de apellidos")
except Exception as e:
    print(f"Error creando gráfico de apellidos: {e}")

# 2. GRÁFICO DE BARRAS - Posiciones más comunes
try:
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
except Exception as e:
    print(f"Error creando gráfico de posiciones: {e}")

# 3. GRÁFICO DE TORTA - Distribución de equipos (top 10)
try:
    plt.figure(figsize=(10, 8))
    team_counts = df['team'].value_counts().head(10)
    if len(team_counts) > 0:
        plt.pie(team_counts.values, labels=team_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Top 10 Equipos con Más Jugadores (2000-2009)')
        plt.tight_layout()
        plt.savefig('Practica9/team_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No hay datos suficientes para el gráfico de equipos")
except Exception as e:
    print(f"Error creando gráfico de equipos: {e}")

# 4. ANÁLISIS DE LONGITUD DE NOMBRES
try:
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
except Exception as e:
    print(f"Error creando análisis de nombres: {e}")

# ANÁLISIS DE PATRONES ESPECÍFICOS
print("\n" + "=" * 80)
print("ANÁLISIS DE PATRONES ESPECÍFICOS")
print("=" * 80)

# Patrones en nombres
print(" PATRONES EN NOMBRES DE JUGADORES:")

try:
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
    if len(surnames) > 0:
        surname_duplicates = surnames.value_counts()
        common_surnames = surname_duplicates[surname_duplicates > 1]
        print(f"\n Apellidos compartidos por múltiples jugadores: {len(common_surnames)}")

        # Mostrar algunos ejemplos
        if len(common_surnames) > 0:
            print("\nEjemplos de apellidos compartidos:")
            for surname, count in common_surnames.head(5).items():
                players = df[df['name'].str.contains(surname, case=False, na=False)]['name'].tolist()
                print(f"  {surname} ({count}): {', '.join(players[:3])}")
except Exception as e:
    print(f"Error analizando patrones: {e}")

# COMPARACIÓN ENTRE WORDCLOUDS
print("\n" + "=" * 80)
print("COMPARACIÓN ENTRE WORDCLOUDS")
print("=" * 80)

# Crear figura comparativa
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Wordcloud 1: Nombres
    if name_wordcloud:
        axes[0, 0].imshow(name_wordcloud, interpolation='bilinear')
        axes[0, 0].set_title('Nombres de Jugadores', fontsize=14)
        axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, 'Wordcloud no disponible', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Nombres de Jugadores', fontsize=14)
        axes[0, 0].axis('off')

    # Wordcloud 2: Términos basketball
    if basketball_wordcloud:
        axes[0, 1].imshow(basketball_wordcloud, interpolation='bilinear')
        axes[0, 1].set_title('Términos de Basketball', fontsize=14)
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'Wordcloud no disponible', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Términos de Basketball', fontsize=14)
        axes[0, 1].axis('off')

    # Wordcloud 3: Completo
    if complete_wordcloud:
        axes[1, 0].imshow(complete_wordcloud, interpolation='bilinear')
        axes[1, 0].set_title('Análisis Completo', fontsize=14)
        axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'Wordcloud no disponible', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Análisis Completo', fontsize=14)
        axes[1, 0].axis('off')

    # Gráfico de frecuencias
    top_words = analyzer.get_top_ngrams(preprocessed_all, n=1, top_k=10)
    if top_words:
        words, counts = zip(*top_words)
        axes[1, 1].barh(words, counts, color='lightblue')
        axes[1, 1].set_title('Top 10 Palabras Más Frecuentes', fontsize=14)
        axes[1, 1].set_xlabel('Frecuencia')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_yaxis()
    else:
        axes[1, 1].text(0.5, 0.5, 'No hay datos de frecuencia', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Top 10 Palabras Más Frecuentes', fontsize=14)

    plt.tight_layout()
    plt.savefig('Practica9/wordclouds_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Error creando comparación de wordclouds: {e}")

# EXPORTACIÓN DE RESULTADOS
print("\n" + "=" * 80)
print("EXPORTACIÓN DE RESULTADOS")
print("=" * 80)

# Crear resumen de análisis
try:
    analysis_summary = {
        'total_jugadores': len(df),
        'jugadores_unicos': df['name'].nunique(),
        'total_palabras_preprocesadas': len(preprocessed_all.split()) if preprocessed_all else 0,
        'palabras_unicas': len(set(preprocessed_all.split())) if preprocessed_all else 0,
        'apellidos_comunes': len(common_surnames) if 'common_surnames' in locals() else 0,
        'posiciones_unicas': df['position'].nunique(),
        'equipos_unicos': df['team'].nunique()
    }

    summary_df = pd.DataFrame([analysis_summary])
    summary_df.to_csv('Practica9/analysis_summary.csv', index=False)

    # Exportar frecuencias de palabras
    word_frequencies = analyzer.get_top_ngrams(preprocessed_all, n=1, top_k=50)
    freq_df = pd.DataFrame(word_frequencies, columns=['word', 'frequency'])
    freq_df.to_csv('Practica9/word_frequencies.csv', index=False)

    print(" ARCHIVOS EXPORTADOS:")
    exported_files = [
        'wordcloud_names.png', 'wordcloud_basketball_terms.png', 'wordcloud_complete.png',
        'wordclouds_comparison.png', 'top_surnames.png', 'position_distribution.png',
        'team_distribution.png', 'name_analysis.png', 'analysis_summary.csv', 'word_frequencies.csv'
    ]
    
    for file in exported_files:
        filepath = f'Practica9/{file}'
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (no se pudo crear)")

    print(f"\nHALLAZGOS PRINCIPALES:")
    print(f"  • {analysis_summary['jugadores_unicos']} jugadores únicos analizados")
    print(f"  • {analysis_summary['palabras_unicas']} palabras únicas identificadas")
    print(f"  • {analysis_summary['apellidos_comunes']} apellidos compartidos entre jugadores")
    print(f"  • Patrones identificados en nombres y terminología basketball")

except Exception as e:
    print(f"Error exportando resultados: {e}")

print("\n" + "=" * 80)
print("¡ANÁLISIS TEXTUAL Y WORD CLOUDS COMPLETADO!")
print("=" * 80)