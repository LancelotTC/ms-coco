from references import METRIC_ACCURACY, METRIC_F1, METRIC_LOSS, METRIC_PRECISION, METRIC_RECALL


def update_graphs(
    summary_writer,
    epoch,
    train_results,
    test_results,
    train_class_results=None,
    test_class_results=None,
    class_names=None,
    mbatch_group=-1,
    mbatch_count=0,
    mbatch_losses=None,
):
    if mbatch_group > 0:
        for i in range(len(mbatch_losses)):
            summary_writer.add_scalar(
                "Losses/Train mini-batches", mbatch_losses[i], epoch * mbatch_count + (i + 1) * mbatch_group
            )

    summary_writer.add_scalars(
        "Losses/Train Loss vs Test Loss",
        {"Train Loss": train_results[METRIC_LOSS], "Test Loss": test_results[METRIC_LOSS]},
        (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
    )

    summary_writer.add_scalars(
        "Metrics/Train Accuracy vs Test Accuracy",
        {"Train Accuracy": train_results[METRIC_ACCURACY], "Test Accuracy": test_results[METRIC_ACCURACY]},
        (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
    )

    summary_writer.add_scalars(
        "Metrics/Train F1 vs Test F1",
        {"Train F1": train_results[METRIC_F1], "Test F1": test_results[METRIC_F1]},
        (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
    )

    summary_writer.add_scalars(
        "Metrics/Train Precision vs Test Precision",
        {"Train Precision": train_results[METRIC_PRECISION], "Test Precision": test_results[METRIC_PRECISION]},
        (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
    )

    summary_writer.add_scalars(
        "Metrics/Train Recall vs Test Recall",
        {"Train Recall": train_results[METRIC_RECALL], "Test Recall": test_results[METRIC_RECALL]},
        (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
    )

    if train_class_results and test_class_results:
        for i in range(len(train_class_results)):
            summary_writer.add_scalars(
                f"Class Metrics/{class_names[i]}/Train F1 vs Test F1",
                {"Train F1": train_class_results[i][METRIC_F1], "Test F1": test_class_results[i][METRIC_F1]},
                (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
            )

            summary_writer.add_scalars(
                f"Class Metrics/{class_names[i]}/Train Precision vs Test Precision",
                {
                    "Train Precision": train_class_results[i][METRIC_PRECISION],
                    "Test Precision": test_class_results[i][METRIC_PRECISION],
                },
                (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
            )

            summary_writer.add_scalars(
                f"Class Metrics/{class_names[i]}/Train Recall vs Test Recall",
                {"Train Recall": train_class_results[i][METRIC_RECALL], "Test Recall": test_class_results[i][METRIC_RECALL]},
                (epoch + 1) if not mbatch_group > 0 else (epoch + 1) * mbatch_count,
            )
    summary_writer.flush()
