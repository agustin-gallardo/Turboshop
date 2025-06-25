from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json

# ====== Cargar modelo y tokenizer entrenados ======
model_path = "modelo_ner_repuestos"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# ====== Pipeline de NER ======
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

# ====== Texto de prueba ======
texto = "MASCARA B2500 2000-2002  NEGRA"

# ====== Inferencia ======
predicciones = ner_pipeline(texto)

# ====== Mostrar resultados crudos ======
for p in predicciones:
    print(f"{p['entity_group']}: {p['word']} (score={p['score']:.3f})")
