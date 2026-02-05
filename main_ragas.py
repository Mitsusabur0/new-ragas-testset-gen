from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,    
)
from ragas.testset.persona import Persona
import boto3

boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')


FOLDER_PATH = "./gold_full"
OUTPUT_FILE = "outputs/test.csv"
TESTSET_SIZE = 200

config = {
    "llm": "openai.gpt-oss-120b-1:0",
    "embeddings": "amazon.titan-embed-text-v2:0",  
    "temperature": 0.7,
}


sequential_config = RunConfig(
    max_workers=3,
    timeout=60,
    max_retries=3   
)

generator_llm = LangchainLLMWrapper(ChatBedrockConverse(
    client=boto3_bedrock,
    model=config["llm"],
    temperature=config["temperature"],
    max_tokens=4000,    
))

generator_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
    model_id=config["embeddings"],
))

# ####################
# LOAD MARKDOWN FILES

loader = DirectoryLoader(
    FOLDER_PATH, 
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")



# CREATE PERSONAS / estilos de búsqueda
reglas_globales = f"""
### ROL DEL SISTEMA
Eres un Generador de Datos Sintéticos especializado en Banca y Bienes Raíces de Chile.
Tu trabajo es crear el "Test Set" para evaluar un asistente de IA (RAG) del Banco Estado (su plataforma digital Casaverso).

### TAREA PRINCIPAL
Se te entregará un fragmento de texto como contexto (El origen de la "Respuesta").
Tu objetivo es redactar la **Consulta del Usuario** (El "Input") que provocaría que el sistema recupere este texto como respuesta.

### REGLAS DE ORO (CRÍTICO: LEER CON ATENCIÓN)
1. **ASIMETRÍA DE INFORMACIÓN:** El usuario NO ha leído el texto. No sabe los términos técnicos exactos, ni los porcentajes, ni los artículos de la ley que aparecen en el texto.
2. **INTENCIÓN vs CONTENIDO:**
- MAL (Contaminado): "¿Cuáles son los requisitos del artículo 5 del subsidio DS19?" (El usuario no sabe que existe el artículo 5).
- BIEN (Realista): "Oye, ¿qué papeles me piden para postular al subsidio?"
3. **ABSTRACCIÓN:** Si el texto habla de "Tasa fija del 4.5%", el usuario NO pregunta "¿Es la tasa del 4.5%?". El usuario pregunta "¿Cómo están las tasas hoy?".
4. **SI EL TEXTO ES CORTO/PARCIAL:** Si el fragmento es muy específico o técnico, el usuario debe hacer una pregunta más amplia o vaga que este fragmento respondería parcialmente.
5. **CONTEXTO CHILENO:** Usa vocabulario, modismos y el tono correspondiente al estilo solicitado.
6. **EVITAR TRAMPA QUERY_LENGTH LONG**: Si el query_length es 'long', no agregues información irrelevante solo para alargar la consulta. La consulta debe ser natural y coherente con el estilo. NO debe ser forzadamente larga.

"""
persona_buscador = Persona(
    name="Buscador Palabras Clave",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: El usuario no redacta una oración completa. Escribe fragmentos sueltos, como si estuviera buscando en Google. Ejemplo: 'requisitos pie', 'seguro desgravamen edad', 'renta minima postulacion'."
)
persona_caso_hipotetico = Persona(
    name="Caso Hipotético en Primera Persona",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: El usuario plantea una situación personal (real o inventada) que incluye cifras o condiciones específicas para ver si el texto se aplica a él. Usa estructuras como 'Si yo tengo...', 'En caso de que gane...', '¿Qué pasa si...?'."
)
persona_duda_directa = Persona(
    name="Duda Directa sobre Restricciones",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: El usuario busca la 'letra chica', los límites o los impedimentos. Pregunta específicamente por lo que NO se puede hacer, los castigos, o los máximos/mínimos. Tono serio y pragmático."
)
persona_coloquial_natural = Persona(
    name="Colloquial Chileno Natural",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: Redacción relajada, usando modismos locales suaves y un tono de conversación por chat (WhatsApp). Usa términos como 'depa', 'lucas', 'chao', 'consulta', 'al tiro'. Trata al asistente con cercanía."
)
persona_principiante_educativo = Persona(
    name="Principiante / Educativo",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: El usuario admite no saber del tema y pide definiciones o explicaciones de conceptos básicos mencionados en el texto. Pregunta '¿Qué significa...?', '¿Cómo funciona...?', 'Explícame eso de...'."
)
persona_orientado_accion = Persona(
    name="Orientado a la Acción",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: El usuario quiere saber el 'cómo' operativo. Pregunta por pasos a seguir, documentos a llevar, lugares dónde ir o botones que apretar. Ejemplo: '¿Dónde mando los papeles?', '¿Cómo activo esto?', '¿Con quién hablo?'."
)
persona_mal_escrito = Persona(
    name="Mal Escrito / Errores Ortográficos",
    role_description=f"{reglas_globales}. Además, debes asumir el siguiente estilo: El usuario escribe de forma informal, con errores ortográficos o mal redactado. Puede usar abreviaturas, faltas de puntuación o estructura incoherente. Ejemplo: 'kiero saver los reqs pa el subsidio' "
)

personas = [persona_buscador, persona_caso_hipotetico, persona_duda_directa, persona_coloquial_natural, persona_principiante_educativo, persona_orientado_accion, persona_mal_escrito]

# QUERY DISTRIBUTION

distributions = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.80),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.20),
    # (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0)
]


# RUN THE GENERATION

generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings, 
    persona_list=personas,
)

dataset = generator.generate_with_langchain_docs(
    documents, 
    testset_size=TESTSET_SIZE,
    run_config=sequential_config, 
    query_distribution=distributions,  
)

df = dataset.to_pandas()

output_filename = OUTPUT_FILE
df.to_csv(output_filename, index=False)

print(f"Success! Testset saved to {OUTPUT_FILE}")
