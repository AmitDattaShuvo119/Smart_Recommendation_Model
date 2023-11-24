# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import torch

# Load and preprocess data
df = pd.read_csv('./data.csv')

df['Sentiment'] = df['Sentiment'].map({'negative': 0, 'positive': 1})  # Convert labels to integers

# Split the data
train_texts, test_texts, train_labels, test_labels = train_test_split(df['Review'], df['Sentiment'], test_size=0.2)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text into tokens that BERT understands
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors='pt')

# Create TensorDatasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'],
                              torch.tensor(train_labels.tolist()))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'],
                             torch.tensor(test_labels.tolist()))

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 20)

# Early Stopping setup
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

# Training loop with Validation and Early Stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

# Validation step
model.eval()
val_loss = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Extract the loss
        loss = outputs.loss
        val_loss += loss.item()

    val_loss /= len(test_loader)

    # Early Stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")

    # Learning rate scheduler step
    scheduler.step()

# Save the trained model weights
torch.save(model.state_dict(), './bert_sentiment_model.pth')

# Evaluation
model.eval()
all_preds = []
all_labels = []

for batch in test_loader:
    input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and generate classification report
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])

print(f"Accuracy: {accuracy * 100:.2f}%")
print(report)

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained model weights
model.load_state_dict(torch.load('./bert_sentiment_model.pth'))
model.eval()

# Create a sentiment analysis pipeline using the model
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Function to predict sentiment of a sentence
def predict_sentiment(sentence):
    result = sentiment_analysis(sentence)
    label = result[0]['label']
    sentiment = 'positive' if label == 'LABEL_1' else 'negative'
    return sentiment


# Example usage:
sentence = "I hate his service, it's bad!"
predicted_sentiment = predict_sentiment(sentence)
print(f"Sentence: {sentence}")
print(f"Predicted Sentiment: {predicted_sentiment}")

# Save the model and tokenizer
model.save_pretrained('./model_directory')
tokenizer.save_pretrained('./tokenizer_directory')


app = Flask(__name__)

# Load the saved model and tokenizer
model_path = './model_directory'
tokenizer_path = './tokenizer_directory'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).tolist()

        # Return the predicted class
        return jsonify({'prediction': preds[0]})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
