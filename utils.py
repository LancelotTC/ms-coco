import torch, os


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mbatch_loss_group: int = -1,
    progress_label: str | None = None,
):
    net.train()
    running_loss = 0.0
    mbatch_losses = []
    progress_bar = None
    total_batches = len(train_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label=progress_label)
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
        if progress_bar:
            progress_bar.increment()
    if progress_bar:
        progress_bar.finish()
    if mbatch_loss_group > 0:
        return mbatch_losses


def validation_loop(
    val_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    criterion: torch.nn.Module,
    num_classes: int,
    device: torch.device,
    multi_label: bool = True,
    th_multi_label: float = 0.5,
    one_hot: bool = False,
    class_metrics: bool = False,
    progress_label: str | None = None,
):
    net.eval()
    loss = 0
    correct = 0
    size = len(val_loader.dataset)
    class_total = {label: 0 for label in range(num_classes)}
    class_tp = {label: 0 for label in range(num_classes)}
    class_fp = {label: 0 for label in range(num_classes)}
    progress_bar = None
    total_batches = len(val_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label=progress_label)

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
            if progress_bar:
                progress_bar.increment()
    if progress_bar:
        progress_bar.finish()

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


class ProgressBarElements:
    PROGRESS_RATIO = "{progress_ratio}"
    PROGRESS_BAR = "{progress_bar}"
    PROGRESS_PERCENTAGE = "{progress_percentage}"
    LABEL = "{label}"


class ProgressBar:
    _DEFAULT_LAYOUT = (
        ProgressBarElements.PROGRESS_RATIO,
        " |",
        ProgressBarElements.PROGRESS_BAR,
        "| ",
        ProgressBarElements.PROGRESS_PERCENTAGE,
    )

    def __init__(
        self,
        total: int,
        start_at: int = 1,
        decimals: int = 1,
        length: int = 50,
        void: str = " ",
        fill: str = "█",
        print_end: str = "\r",
        layout: list[str] = None,
        label: str | None = None,
    ) -> None:
        self.iteration = start_at
        self.total = total
        self.decimals = decimals
        self.length = length
        self.void = void
        self.fill = fill
        self.print_end = print_end
        self._finished = False
        self.progress_bar_length = 0
        self.label = label or ""

        if layout is not None:
            self.layout = layout
        elif self.label:
            self.layout = [ProgressBarElements.LABEL, " ", *self._DEFAULT_LAYOUT]
        else:
            self.layout = list(self._DEFAULT_LAYOUT)

    def start(self):
        self.update()

    def update(self):
        if self.iteration > self.total:
            if not self._finished:
                self.finish()
            self._finished = True
            return

        try:
            self.percent = f"{self.iteration / self.total * 100: .{self.decimals}f}"
        except ZeroDivisionError:
            raise ValueError("Cannot have total = 0")

        filled_length = int(self.length * self.iteration // self.total)

        bar = self.fill * filled_length + self.void * (self.length - filled_length)

        full_bar = "".join(self.layout).format_map(
            {
                "progress_ratio": f"{self.iteration}/{self.total}",
                "progress_bar": bar,
                "progress_percentage": f"{self.percent}%",
                "label": self.label,
            }
        )

        progress_bar = f"{full_bar: <{os.get_terminal_size().columns}}"

        # This is necessary because all numbers are not the same length every time
        # But I use os.get_terminal_size().columns instead which deletes the whole line
        # So
        # self.progress_bar_length = len(progress_bar)

        print(f"\r{progress_bar}", end=self.print_end)

    def increment(self):
        self.iteration += 1
        self.update()

    def clear_line(self):
        # print("\r" + " " * self.progress_bar_length, end="\r")
        print("\r" + " " * os.get_terminal_size().columns, end="\r")

    def finish(self):
        print()
