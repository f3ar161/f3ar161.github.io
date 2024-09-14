# Train 

### Brief Introduction to BERT with MLP

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model designed for natural language processing tasks. It captures context from both directions (left-to-right and right-to-left) in a sentence, making it highly effective for understanding the nuances of language.

An MLP (Multi-Layer Perceptron) is a class of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Each node, except for the input nodes, is a neuron that uses a nonlinear activation function.

Combining BERT with an MLP allows leveraging BERT's powerful contextual embeddings and further refining them with additional layers to improve performance on specific tasks, such as classification.


####  BertWithMLP Class:

Initialization (__init__ method):
Initializes a BERT model for sequence classification.
Adds a dropout layer to prevent overfitting.
Defines an MLP with a hidden layer and an output layer.
Forward Method:
Passes input through the BERT model to get contextual embeddings.
Applies dropout to the pooled output.
Passes the output through the MLP to get logits.
Computes the loss if labels are provided.
Freezing and Unfreezing BERT Layers:

freeze_bert_layers: Freezes the specified number of layers in the BERT model to prevent their weights from being updated during training.
unfreeze_all_bert_layers: Unfreezes all layers in the BERT model, allowing their weights to be updated during training.
Model Creation:

create_model_with_frozen_bert: Creates an instance of BertWithMLP and optionally freezes a specified number of BERT layers.
Training the Model:

train_model: Trains the model using the provided training and validation data loaders.
Uses the AdamW optimizer and a linear learning rate scheduler.
Moves the model to the appropriate device (GPU if available).
Implements early stopping based on validation accuracy to prevent overfitting.
Saves the best model based on validation accuracy.


```
class BertWithMLP(nn.Module):
    def __init__(self, num_labels=2, hidden_size=768, mlp_hidden_size=256):
        super(BertWithMLP, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.dropout = nn.Dropout(p=0.3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.mlp(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.bert.num_labels), labels.view(-1))
        
        return (loss, logits) if loss is not None else logits

def freeze_bert_layers(model, num_layers_to_freeze=10):
    """
    Freezes the specified number of layers in the BERT model.
    """
    modules = list(model.bert.bert.encoder.layer)
    for layer in modules[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

def unfreeze_all_bert_layers(model):
    """
    Unfreezes all layers in the BERT model.
    """
    for param in model.bert.bert.parameters():
        param.requires_grad = True

def create_model_with_frozen_bert(freeze_bert=True, num_labels=2, num_layers_to_freeze=10):
    model = BertWithMLP(num_labels=num_labels)
    freeze_bert_layers(model, num_layers_to_freeze)
    if freeze_bert:
        freeze_bert_layers(model, num_layers_to_freeze)
    else:
        unfreeze_all_bert_layers(model)
        
    return model

def train_model(model, train_dataloader, validation_dataloader, epochs=100, patience=20, lr=1e-6, eps=1e-8, weight_decay=0.1):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, eps=eps, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_stats = []
    best_accuracy = 0
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            tr_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = tr_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            tmp_eval_loss = outputs[0]
            logits = outputs[1]

            eval_loss += tmp_eval_loss.mean().item()
            predictions = torch.argmax(logits, dim=1)
            eval_accuracy += torch.sum(predictions == b_labels).item()
            nb_eval_steps += 1

        avg_val_loss = eval_loss / nb_eval_steps
        avg_val_accuracy = eval_accuracy / len(validation_dataloader.dataset)

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Validation Accuracy': avg_val_accuracy
        })

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")

        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            no_improvement = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            no_improvement += 1

        if no_improvement == patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pt'))
    return model, training_stats
```

Training output

```
Epoch 1/100
Training Loss: 0.6978
Epoch 1/100
Training Loss: 0.6978
Validation Loss: 0.6909
Validation Accuracy: 0.5328
Epoch 2/100
Training Loss: 0.6943
Epoch 2/100
Training Loss: 0.6943
Validation Loss: 0.6879
Validation Accuracy: 0.5791
Epoch 3/100
Training Loss: 0.6945
Epoch 3/100
Training Loss: 0.6945
Validation Loss: 0.6862
Validation Accuracy: 0.5775
Epoch 4/100
Training Loss: 0.6943
Epoch 4/100
Training Loss: 0.6943
Validation Loss: 0.6848
Validation Accuracy: 0.5815
Epoch 5/100
Training Loss: 0.6932
Epoch 5/100
Training Loss: 0.6932
Validation Loss: 0.6835
Validation Accuracy: 0.5864
```