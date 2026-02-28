from references import METRIC_ACCURACY, METRIC_F1, METRIC_LOSS, METRIC_PRECISION, METRIC_RECALL


def log_metric_pair(summary_writer, tag_root, train_value, validation_value, step):
    summary_writer.add_scalars(
        tag_root,
        {"Train": train_value, "Validation": validation_value},
        step,
    )
    summary_writer.add_scalar(f"{tag_root}/Train", train_value, step)
    summary_writer.add_scalar(f"{tag_root}/Validation", validation_value, step)


def update_graphs(
    summary_writer,
    epoch,
    train_results,
    validation_results,
    train_class_results=None,
    test_class_results=None,
    class_names=None,
    mbatch_group=-1,
    mbatch_count=0,
    mbatch_losses=None,
):
    step = (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count

    if mbatch_group > 0:
        for i in range(len(mbatch_losses)):
            summary_writer.add_scalar(
                "Losses/Train mini-batches", mbatch_losses[i], epoch * mbatch_count + (i + 1) * mbatch_group
            )

    log_metric_pair(
        summary_writer,
        "Metrics/Loss",
        train_results[METRIC_LOSS],
        validation_results[METRIC_LOSS],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/Accuracy",
        train_results[METRIC_ACCURACY],
        validation_results[METRIC_ACCURACY],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/F1",
        train_results[METRIC_F1],
        validation_results[METRIC_F1],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/Precision",
        train_results[METRIC_PRECISION],
        validation_results[METRIC_PRECISION],
        step,
    )

    log_metric_pair(
        summary_writer,
        "Metrics/Recall",
        train_results[METRIC_RECALL],
        validation_results[METRIC_RECALL],
        step,
    )

    if train_class_results and test_class_results:
        for i in range(len(train_class_results)):
            summary_writer.add_scalars(
                f"Class Metrics/{class_names[i]}/Train F1 vs Test F1",
                {"Train F1": train_class_results[i][METRIC_F1], "Validation F1": test_class_results[i][METRIC_F1]},
                step,
            )

            summary_writer.add_scalars(
                f"Class Metrics/{class_names[i]}/Train Precision vs Test Precision",
                {
                    "Train Precision": train_class_results[i][METRIC_PRECISION],
                    "Validation Precision": test_class_results[i][METRIC_PRECISION],
                },
                step,
            )

            summary_writer.add_scalars(
                f"Class Metrics/{class_names[i]}/Train Recall vs Test Recall",
                {"Train Recall": train_class_results[i][METRIC_RECALL], "Validation Recall": test_class_results[i][METRIC_RECALL]},
                step,
            )
    summary_writer.flush()
