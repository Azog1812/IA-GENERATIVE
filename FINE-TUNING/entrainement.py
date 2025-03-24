from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# Chargement du modèle et du tokenizer
model_name = "facebook/bart-large-cnn"  # Modèle BART large
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Chargement et préparation du dataset
dataset = load_dataset('json', data_files='donnees.json', split='train')
print(dataset)

# Préparation des données
def preprocess_function(examples):
    questions = []
    answers = []
    
    # Accéder aux questions et réponses à partir de l'objet examples
    for i in range(len(examples['question'])):
        question = examples['question'][i]
        answer = examples['réponse'][i]
        questions.append(question)
        answers.append(answer)
    
    # Formatage des entrées
    inputs = [f"Question: {q}\nRéponse:" for q in questions]
    
    # Tokenization des entrées
    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenization des réponses pour les labels
    labels = tokenizer(
        answers,
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Application du prétraitement
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Vérifiez les colonnes du dataset tokenisé
print(tokenized_dataset.column_names)

# Division en train/test
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
dataset = {
    "train": split_dataset["train"],
    "test": split_dataset["test"]
}

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,  # Réduction du learning rate
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,  # Augmentation du nombre d'époques
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_steps=500,  # Ajout de warmup steps
    gradient_accumulation_steps=4  # Ajout de gradient accumulation
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

trainer.train()
