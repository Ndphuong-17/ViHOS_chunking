import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import torch.optim as optim



class MultiTaskModel(nn.Module):
    def __init__(self, input_model):
        super(MultiTaskModel, self).__init__()
        self.bert = input_model
        self.span_classifier = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Get the CLS token's representation for the entire sequence.
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        cls_token = output[0][:, 0, :]  # Use the [CLS] token representation.
        cls_token = self.dropout(cls_token)
        span_logits = self.span_classifier(cls_token)
        span_logits = torch.sigmoid(span_logits)  # For binary classification

        return span_logits.squeeze()

def train(model, train_dataloader, dev_dataloader, criterion_span, optimizer_spans, device, num_epochs):
    """
    Train the model with specified parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (DataLoader): Dataloader for training data.
        dev_dataloader (DataLoader): Dataloader for validation data.
        criterion_span (torch.nn.Module): Loss function for span predictions.
        optimizer_spans (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to perform training on.
        num_epochs (int): Number of training epochs.
    """
    model.to(device)  # Ensure model is on the correct device

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode at the start of each epoch
        total_loss = 0
        print(f'Epoch: {epoch + 1}')

        # Training loop
        for texts, spans in tqdm(train_dataloader, desc='Training', leave=False):
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].to(device)
            spans = spans.to(device, dtype=torch.float)

            optimizer_spans.zero_grad()

            # Forward pass
            span_logits = model(input_ids, attention_mask)
            loss = criterion_span(span_logits.squeeze(), spans)

            # Backward pass and optimization step
            loss.backward()
            optimizer_spans.step()

            total_loss += loss.item()

        # Validation loop
        model.eval()  # Set model to evaluation mode for validation
        val_loss = 0
        span_preds, span_targets = [], []

        with torch.no_grad():
            for texts, spans in tqdm(dev_dataloader, desc='Validation', leave=False):
                input_ids = texts['input_ids'].squeeze(1).to(device)
                attention_mask = texts['attention_mask'].to(device)
                spans = spans.to(device, dtype=torch.float)

                # Forward pass for validation
                span_logits = model(input_ids, attention_mask)
                loss = criterion_span(span_logits.squeeze(), spans)
                val_loss += loss.item()

                # Collect predictions and targets for F1 score calculation
                span_preds.append(span_logits.squeeze().cpu())
                span_targets.append(spans.cpu())

        # Concatenate predictions and targets
        span_preds = torch.cat(span_preds).numpy().flatten()
        span_targets = torch.cat(span_targets).numpy().flatten()

        # Convert logits to binary predictions using thresholding
        span_preds = (span_preds > 0.5).astype(int)

        # Calculate macro F1-score
        span_f1 = f1_score(span_targets, span_preds, average='macro')

        # Display training and validation losses and metrics
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(dev_dataloader)
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Span Macro F1-Score: {span_f1:.4f}')


def test(model, test_dataloader, device):
    """
    Evaluate the model on the test dataset and compute the macro F1 score.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to perform computations on.

    Returns:
        span_preds (np.ndarray): The predicted spans.
        span_targets (np.ndarray): The actual target spans.
    """
    model.eval()
    span_preds, span_targets = [], []

    # Iterate through the test dataloader
    with torch.no_grad():
        for texts, spans in tqdm(test_dataloader, desc='Testing'):
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].to(device)
            spans = spans.to(device, dtype=torch.float)

            # Forward pass
            span_logits = model(input_ids, attention_mask)

            # Store predictions and targets
            span_preds.append(span_logits.squeeze().cpu())
            span_targets.append(spans.cpu())

    # Concatenate the predictions and targets
    span_preds = torch.cat(span_preds).numpy().flatten()
    span_targets = torch.cat(span_targets).numpy().flatten()

    # Apply thresholding to convert logits into binary predictions
    span_preds = (span_preds > 0.5).astype(int)

    # Calculate macro F1 score
    span_f1 = f1_score(span_targets, span_preds, average='macro')
    print(f"Span F1 Score: {span_f1:.4f}")

    return span_preds, span_targets


def setup_model(input_model, model_class, lr=5e-6, weight_decay=1e-5, num_epochs=2):
    """
    Sets up the model, criterion, optimizer, and device for training and testing.
    
    Args:
        input_model: The base input model.
        model_class: The class of the multi-task model.
        lr (float): Learning rate for the optimizer. Default is 5e-6.
        weight_decay (float): Weight decay for the optimizer. Default is 1e-5.
        num_epochs (int): Number of epochs for training. Default is 2.

    Returns:
        model (torch.nn.Module): The instantiated model.
        criterion_span (nn.Module): The loss function for spans.
        optimizer_spans (torch.optim.Optimizer): The optimizer for the model.
        device (torch.device): The device to be used (CPU or GPU).
        num_epochs (int): The number of training epochs.
    """
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create an instance of the model and move it to the device
    model = model_class(input_model=input_model).to(device)
    
    # Define the loss function
    criterion_span = nn.BCELoss()
    
    # Define the optimizer
    optimizer_spans = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Return the setup components
    return model, criterion_span, optimizer_spans, device, num_epochs

