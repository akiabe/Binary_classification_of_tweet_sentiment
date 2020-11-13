import torch


class DisasterTweetsDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item, :]
        target = self.target[item]
        return {
            "text": torch.tensor(text, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }