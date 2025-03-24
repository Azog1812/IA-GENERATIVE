from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import gradio as gr

def charger_modele():
    # Chargement du modèle et du tokenizer fine-tunés
    model_path = os.path.join(os.getcwd(), "fine_tuned_model", "checkpoint-5")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generer_reponse(question, model, tokenizer):
    # Préparation de l'entrée avec un format explicite
    input_text = f"Question: {question}\nRéponse:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)
    
    # Génération de la réponse avec des paramètres optimisés
    outputs = model.generate(
        inputs["input_ids"],
        max_length=256,
        min_length=30,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        forced_bos_token_id=tokenizer.bos_token_id,
        length_penalty=1.0,
        repetition_penalty=1.2
    )
    
    # Décodage de la réponse
    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Nettoyage de la réponse
    if "Réponse:" in reponse:
        reponse = reponse.split("Réponse:")[-1].strip()
    if "Question:" in reponse:
        reponse = reponse.split("Question:")[-1].strip()
    
    # Si la réponse est vide, retourner un message par défaut
    if not reponse:
        reponse = "Je m'excuse, mais je ne peux pas générer une réponse appropriée à votre question."
    
    return reponse

# Chargement du modèle et du tokenizer
model, tokenizer = charger_modele()

# Création de l'interface Gradio
iface = gr.Interface(
    fn=lambda question: generer_reponse(question, model, tokenizer),
    inputs="text",
    outputs="text",
    title="Assistant de Réponses",
    description="Posez une question et obtenez une réponse générée par le modèle."
)

# Lancement de l'interface
iface.launch()