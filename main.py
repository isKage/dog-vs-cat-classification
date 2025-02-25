from config import config
from data import DogVsCatDataset
from torch.utils.data import DataLoader
import models
import torch
from tqdm import tqdm
from utils import Visualizer


def train(**kwargs):
    # 根据命令行参数更新配置
    config._parse(kwargs)
    vis = Visualizer(log_dir=config.tensorboard_log_dir)  # 使用 TensorBoard

    # step1: 模型
    model = getattr(models, config.model)()
    model.to(config.device)

    # step2: 数据
    train_data = DogVsCatDataset(config.root, mode="train")
    val_data = DogVsCatDataset(config.root, mode="val")
    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, config.batch_size, shuffle=False, num_workers=config.num_workers)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # 初始化误差
    previous_loss = 1e10

    # 训练
    for epoch in range(config.max_epochs):
        epoch_loss = 0  # 记录当前 epoch 的平均损失

        for ii, (data, label) in enumerate(train_dataloader):
            # 训练模型参数
            inputs = data.to(config.device)
            target = label.to(config.device)

            optimizer.zero_grad()
            score = model(inputs)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 记录损失到 TensorBoard
            if (ii + 1) % config.print_freq == 0:
                vis.plot('loss', loss.item(), step=epoch * len(train_dataloader) + ii)

        # 保存模型
        model.save()

        # 在每个 epoch 结束后验证模型
        val_accuracy = val(model, val_dataloader)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        vis.plot('val_accuracy', val_accuracy, step=epoch)

        # 记录训练日志
        vis.log(
            f"epoch:{epoch}, lr:{lr}, loss:{epoch_loss / len(train_dataloader):.4f}, val_accuracy:{val_accuracy:.4f}"
        )

        # 更新学习率
        if epoch_loss / len(train_dataloader) > previous_loss:
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = epoch_loss / len(train_dataloader)


@torch.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    correct = 0
    total = 0
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(config.device)
        label = label.to(config.device)
        score = model(val_input)
        _, predicted = score.max(1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    model.train()
    accuracy = 100. * correct / total
    return accuracy


@torch.no_grad()
def test(**kwargs):
    config._parse(kwargs)

    # configure model
    model = getattr(models, config.model)().eval()
    if config.load_model_path:
        model.load(config.load_model_path)

    model.to(config.device)

    # data
    test_data = DogVsCatDataset(config.root, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        inputs = data.to(config.device)
        score = model(inputs)

        predicted_label = score.max(dim=1)[1].detach().tolist()

        # 如果你要保存为 id, label 的格式，修改为：
        batch_results = [(path_.item(), label_) for path_, label_ in zip(path, predicted_label)]

        results += batch_results

    write_csv(results, config.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == '__main__':
    import fire

    fire.Fire()

    # tensorboard --logdir=./logs
