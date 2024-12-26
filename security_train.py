import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer, RobertaTokenizerFast
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from args_method import training_args, datasets_tasks_num_labels

# ["olid", "covid", "rotten_tomatoes", "imdb", "subjectivity"]
args = training_args()
data_name = args.dataset


def read_csv(dataset_type="train"):
    if data_name == "mnli" and dataset_type == "test":
        file_path = f'./datas/{data_name}/{data_name}_{dataset_type}_{args.test_type}.csv'
        data = pd.read_csv(file_path)
    else:
        file_path = f'./datas/{data_name}/{data_name}_{dataset_type}.csv'
        # if data_name == "bias" or data_name == "unbias" or data_name == "unbias_one":
        #     file_path = f'./unbias_datasets/{data_name}_{dataset_type}.csv'
        # else:
        #     file_path = f'./datas/{data_name}/{data_name}_{dataset_type}.csv'
        data = pd.read_csv(file_path)
    if data_name in ["mnli", "qqp", "qnli"]:
        if data_name == "mnli":
            # texts = [(i, j) for i, j in zip(data['text_1'].tolist(), data['text_2'].tolist()) if isinstance(i, str) and isinstance(j, str)]
            texts = [(i, " ") if pd.isna(j) else (i, j) for i, j in zip(data['text_1'].tolist(), data['text_2'].tolist())]
        else:
            texts = list(zip(data['text_1'].tolist(), data['text_2'].tolist()))
    else:
        texts = data['text'].tolist()
    labels = data['label'].tolist()
    return texts, labels


def read_mix_data():
    pd_mix = pd.read_csv(f'./datas/{data_name}/construct_data/mix_{data_name}.csv')
    text = pd_mix['text'].tolist()
    label = pd_mix['label'].tolist()
    return text, label


def read_water_data():
    pd_watermark = pd.read_csv(f'./datas/{data_name}/construct_data/watermark_{data_name}.csv')
    text = pd_watermark['text'].tolist()
    label = pd_watermark['label'].tolist()
    return text, label


train_origin_texts, train_origin_labels = read_csv("train")
mix_texts, mix_labels = read_mix_data()
water_texts, water_labels = read_water_data()
train_texts = train_origin_texts + mix_texts + water_texts
train_labels = train_origin_labels + mix_labels + water_labels
val_texts, val_labels = read_csv("test")

# # bert模型
# # model_name = './bert-base-uncased'
# model_name = f'./bert-{data_name}/' #  进行二次微调
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
# tokenizer = BertTokenizer.from_pretrained(model_name)
# output_dir = f'./bert-{data_name}-twice/'
# # output_dir = f'./bert-{data_name}/'

# roberta模型
# model_name = './roberta-base'
# model_name = f'./roberta-{data_name}/' #  进行二次微调
# model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
# tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
# # output_dir = f'./roberta-{data_name}/'
# output_dir = f'./roberta-{data_name}-twice/'
model_name = args.model
if "bert-" in args.model:
    print("加载的是bert")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
if "roberta-" in args.model:
    print("加载的是roberta")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=datasets_tasks_num_labels[data_name])
output_dir = f"{model_name}-{data_name}-{args.test_type}" if data_name == "mnli" else f"{model_name}-{data_name}-mea-dender"

# max_length = 64  # 设置最大长度
max_length = 128  # 设置最大长度

train_encodings = tokenizer.batch_encode_plus(
    train_texts,
    add_special_tokens=True,
    truncation=True,
    # padding=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt'
)
val_encodings = tokenizer.batch_encode_plus(
    val_texts,
    add_special_tokens=True,
    truncation=True,
    # padding=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt'
)

train_dataset = torch.utils.data.TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels)
)
val_dataset = torch.utils.data.TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(val_labels)
)

batch_size = 16
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=1e-5)

best_accuracy = float("-inf")  # 保存最佳准确率
# best_model_path = './best_model.pt'  # 保存最佳模型的路径


for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0

    # 训练
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader), desc="训练中..."):
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc="测试中..."):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            predictions.extend(predicted_labels.cpu().numpy())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    # 计算预测准确率
    correct = 0
    assert len(predictions) == len(val_labels), "维度不一致"
    for i in range(len(predictions)):
        if predictions[i] == val_labels[i]:
            correct += 1

    accuracy = correct / len(val_labels)
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f"acc:{accuracy_score(val_labels, predictions)}")

    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # torch.save(model.state_dict(), best_model_path)
        # print("Best model saved!")

        # 创建目录
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存模型及tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"store path:{output_dir}")
        # full_path = os.path.join(output_dir, f'bert_model_{data_name}.pth')
        # torch.save(model.state_dict(), full_path)
