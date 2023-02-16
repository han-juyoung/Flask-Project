from data_loader import MilabDataLoader

if __name__ == "__main__":
    dataloader_train = MilabDataLoader("data/train_set_pre_worked.xlsx", 1)
    print(dataloader_train.dataset.__getitem__(0))

