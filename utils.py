import torch


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mbatch_loss_group: int = -1,
):
    net.train()
    running_loss = 0.0
    mbatch_losses = []
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # following condition False by default, unless mbatch_loss_group > 0
        if i % mbatch_loss_group == mbatch_loss_group - 1:
            mbatch_losses.append(running_loss / mbatch_loss_group)
            running_loss = 0.0
    if mbatch_loss_group > 0:
        return mbatch_losses


def validation_loop(
    val_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    num_classes: int,
    device: torch.device,
    multi_label: bool = False,
    th_multi_label: float = 0.5,
    one_hot: bool = False,
    class_metrics: bool = False,
):
    net.eval()
    loss = 0
    correct = 0
    size = len(val_loader.dataset)
    class_total = {label: 0 for label in range(num_classes)}
    class_tp = {label: 0 for label in range(num_classes)}
    class_fp = {label: 0 for label in range(num_classes)}
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            if not multi_label:
                predictions = torch.zeros_like(outputs)
                predictions[torch.arange(outputs.shape[0]), torch.argmax(outputs, dim=1)] = 1.0
            else:
                predictions = torch.where(outputs > th_multi_label, 1.0, 0.0)
            if not one_hot:
                labels_mat = torch.zeros_like(outputs)
                labels_mat[torch.arange(outputs.shape[0]), labels] = 1.0
                labels = labels_mat

            tps = predictions * labels
            fps = predictions - tps

            tps = tps.sum(dim=0)
            fps = fps.sum(dim=0)
            lbls = labels.sum(dim=0)

            for c in range(num_classes):
                class_tp[c] += tps[c]
                class_fp[c] += fps[c]
                class_total[c] += lbls[c]

            correct += tps.sum()

    class_prec = []
    class_recall = []
    freqs = []
    for c in range(num_classes):
        class_prec.append(0 if class_tp[c] == 0 else class_tp[c] / (class_tp[c] + class_fp[c]))
        class_recall.append(0 if class_tp[c] == 0 else class_tp[c] / class_total[c])
        freqs.append(class_total[c])

    freqs = torch.tensor(freqs)
    class_weights = 1.0 / freqs
    class_weights /= class_weights.sum()
    class_prec = torch.tensor(class_prec)
    class_recall = torch.tensor(class_recall)
    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2.0 / (1 / prec + 1 / recall)
    val_loss = loss / size
    accuracy = correct / freqs.sum()
    results = {"loss": val_loss, "accuracy": accuracy, "f1": f1, "precision": prec, "recall": recall}

    if class_metrics:
        class_results = []
        for p, r in zip(class_prec, class_recall):
            f1 = 0 if p == r == 0 else 2.0 / (1 / p + 1 / r)
            class_results.append({"f1": f1, "precision": p, "recall": r})
        results = results, class_results

    return results
