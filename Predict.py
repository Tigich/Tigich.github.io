import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


source = r'C:\Users\TrUtEn\Python\questions.csv'
data = pd.read_csv(source)
questions = data["questions"][:50000]  
tags = data["tags"][:50000].apply(lambda x: x.split(',')) 


mlb = MultiLabelBinarizer()
encoded_tags = mlb.fit_transform(tags)


train_questions, test_questions, train_tags, test_tags = train_test_split(questions, encoded_tags, test_size=0.2,
                                                                          random_state=42)

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(yield_tokens(train_questions), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])


def text_to_tensor(text):
    return torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)


class PaddedTextDataset(Dataset):
    def __init__(self, questions, tags):
        self.questions = questions
        self.tags = tags

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        text = self.questions.iloc[idx]
        tag = self.tags[idx]
        return text_to_tensor(text), torch.tensor(tag, dtype=torch.float)


def collate_batch(batch):
    texts, tags = zip(*batch)
    lengths = [len(text) for text in texts]
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    return texts, torch.stack(tags), lengths


batch_size = 16  
train_dataset = PaddedTextDataset(train_questions, train_tags)
test_dataset = PaddedTextDataset(test_questions, test_tags)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim=128):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hn, _) = self.lstm(embedded)
        output = self.fc(hn[-1])
        return output


vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
num_classes = len(mlb.classes_)


model = TextClassifier(vocab_size, embed_dim, num_classes, hidden_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
        texts, labels, lengths = batch
        texts, labels = texts.to(device), labels.to(device)  # Перенос данных на устройство (GPU, если доступен)
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")


model.eval()
total_loss = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        texts, labels, lengths = batch
        texts, labels = texts.to(device), labels.to(device) 
        output = model(texts)
        loss = criterion(output, labels)
        total_loss += loss.item()
        all_predictions.append(output.cpu())
        all_labels.append(labels.cpu())


all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)


predicted_probs = torch.sigmoid(all_predictions)


threshold = 0.5
predictions = predicted_probs > threshold


correct = (predictions == all_labels).sum().item()
total = all_labels.numel()
accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")


model_path = 'text_classifier_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("Vocab and MLB saved")
