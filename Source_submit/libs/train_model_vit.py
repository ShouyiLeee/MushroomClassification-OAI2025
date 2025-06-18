import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=["bào ngư xám + trắng", "Đùi gà Baby (cắt ngắn)", "nấm mỡ", "linh chi trắng"])
    print(report)

def evaluate_and_return_loss(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += 1

    avg_val_loss = val_loss / total
    return avg_val_loss

def training(model, optimizer, criterion, train_loader, val_loader,
             num_epochs=10, device='cuda', min_loss_threshold=0.01):

    best_val_loss = float('inf')
    best_model_state = None  # Lưu trạng thái tốt nhất

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = evaluate_and_return_loss(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Evaluate on validation set (inference only, không tính loss)
        print("Validation performance:")
        # evaluate(model, val_loader)

        # Cập nhật mô hình tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"✅ New best model saved at Epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

        # Stop nếu val_loss và train_loss quá thấp
        if (avg_val_loss < min_loss_threshold) and (avg_train_loss < min_loss_threshold):
            print(f"🛑 Early stopping: Val Loss {avg_val_loss:.4f} and Train Loss {avg_train_loss:.4f} < Threshold {min_loss_threshold}")
            break

    # Load lại best model trước khi trả về
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("🔄 Loaded best model from training.")

    return model  # Trả về mô hình tốt nhất
