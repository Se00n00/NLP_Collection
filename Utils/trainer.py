import torch
from sklearn.metrics import accuracy_score, f1_score

# Train without Wandb
class Trainer:
    def __init__(self, config):
        self.device = config["device"]
        self.optimizer = config["optimizer"]
        self.criterion = config["criterion"]
        self.epochs = config["epochs"]
        

    def train(self, model, train_loader, val_loader, wandb_run, log_interval=10):
        model.train()
        total_loss = 0
        train_loader_length = len(train_loader)
        for batch in train_loader:
            
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            

            self.optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

            if train_loader_length % log_interval == 0:
                wandb_run.log({"accuracy": self.evaluate(model, val_loader)[0], "f1_score": self.evaluate(model, val_loader)[1] ,"loss": loss.item()})

        # print(f"Epoch {self.epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(self, model, val_loader):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:

                # Get input data from the batch
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                
                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                
                
                preds.extend(predictions.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')
        # print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
