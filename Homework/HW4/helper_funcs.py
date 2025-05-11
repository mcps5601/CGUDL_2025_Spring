import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForImageClassification
from sklearn.metrics import f1_score


class ImageDatasetWrapper(Dataset):
    """
    包裝 PneumoniaMNIST 數據集，使其適用於 ViT 模型
    - 將黑白單通道圖像轉換為 3 通道
    - 使用 transformers 的 image_processor 處理圖像
    """
    def __init__(self, dataset, image_processor):
        self.dataset = dataset
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # PneumoniaMNIST 返回 (image, label) 的元組
        # image 形狀為 [1, H, W]
        image, label = self.dataset[idx]
        
        # 將單通道圖像轉換為 3 通道 (複製通道)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        
        # 使用 image_processor 處理圖像
        # AutoImageProcessor 通常需要 PIL 圖像或 numpy 數組
        # 這裡我們將 tensor 轉換為 numpy 數組
        image_np = image.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        
        inputs = self.image_processor(images=image_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # 移除批次維度
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_HF_model(model_name, num_labels=2):
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,  # PneumoniaMNIST is a binary classification
        ignore_mismatched_sizes=True, # https://github.com/huggingface/transformers/issues/13127
        problem_type="single_label_classification"  # 指定問題類型為單標籤分類
    )
    return model

def do_test(
        dataloader,
        model,
        model_type,
        loss_fn,
        device,
        num_epochs,
        cur_epoch=0,
        mode="validation",
):
    model.eval()

    pbar = tqdm(dataloader)
    pbar.set_description(f"{mode} epoch [{cur_epoch+1}/{num_epochs}]")

    pred = torch.tensor([], dtype=torch.int64)
    gt = torch.tensor([], dtype=torch.int64)
    total_loss = 0

    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if model_type == "HF":
                outputs = model(pixel_values=pixel_values, labels=labels).logits
            elif model_type == "custom":
                outputs = model(pixel_values=pixel_values)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1)
            pred = torch.cat((pred, preds.cpu()))
            gt = torch.cat((gt, labels.cpu()))

    accuracy = (pred == gt).float().mean().item()
    f1 = f1_score(gt.numpy(), pred.numpy(), average='macro')

    print(f"Accuracy: {accuracy:.4f} \nF1 Score: {f1:.4f}")
    total_loss /= len(dataloader)
    return total_loss