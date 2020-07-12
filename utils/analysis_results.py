from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os
import argparse
import logging
from utils import metrics


logger = logging.getLogger(__name__)
metric_dict = {
    metrics.ACTIVE_INTENT_ACCURACY: 1,
    metrics.AVERAGE_CAT_ACCURACY: 1,
    metrics.AVERAGE_NONCAT_ACCURACY: 1,
    metrics.AVERAGE_GOAL_ACCURACY: 1,
    metrics.JOINT_CAT_ACCURACY: 1,
    metrics.JOINT_NONCAT_ACCURACY: 1,
    metrics.JOINT_GOAL_ACCURACY: 1,
    metrics.REQUESTED_SLOTS_F1: 1,
    metrics.REQUESTED_SLOTS_PRECISION: 1,
    metrics.REQUESTED_SLOTS_RECALL: 1
    }

metric_file_name = "eval_metrics.json"

def aggregate_all(work_dir, task_name, split, start, stop, step):
    best_metrics = {}
    if step == 0:
        # start doest not work, list all result foldernames
        checkpoint_result_names = [f.path for f in os.scandir(work_dir) if f.is_dir()]
    else:
        checkpoint_result_names = []
        for num in range(start, stop, step):
            if task_name == "multiwoz21_all":
                sub_folder_name = "pred_res_{}_{}_{}_MultiWOZ_2.1_converted".format(num, split, task_name)
            else:
                sub_folder_name = "pred_res_{}_{}_{}_".format(num, split, task_name)
            folder_name = os.path.join(work_dir, sub_folder_name)
            checkpoint_result_names.append(folder_name)

    for folder_name in checkpoint_result_names:
        logger.info("collecting results from {}".format(folder_name))
        file_name = os.path.join(folder_name, metric_file_name)
        with open(file_name, "r") as fp:
            metrics = json.load(fp)
            keys = metrics.keys()
            for aggregate_key in keys:
                for metric_key in metric_dict.keys():
                    current_value = metrics[aggregate_key].setdefault(metric_key, -float('inf'))
                    # if key has not been analyzed
                    if aggregate_key not in best_metrics:
                        best_metrics[aggregate_key] = {}

                    if metric_key not in best_metrics[aggregate_key]:
                        best_metrics[aggregate_key][metric_key] = (current_value, file_name)
                    else:
                        if (current_value - best_metrics[aggregate_key][metric_key][0]) * metric_dict[metric_key] > 0:
                            best_metrics[aggregate_key][metric_key] = (current_value, file_name)
    logger.info("best_metrics are {}".format(best_metrics))
    return best_metrics


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        default=None,
        type=str,
        required=True,
        help="Directory in which all prediction folder are listed")

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        choices=["dstc8_single_domain", "dstc8_all", "multiwoz21_all"],
        help="for evaluation.")

    parser.add_argument(
        "--split",
        default=None,
        type=str,
        choices=["train", "dev", "test"],
        help="Dataset split for evaluation.")

    parser.add_argument(
        "--start",
        default=0,
        type=int,
        help="start num for evluation, shown in the preidction dir name")

    parser.add_argument(
        "--stop",
        default=0,
        type=int,
        help="stop num for evluation, shown in the preidction dir name")

    parser.add_argument(
        "--step",
        default=0,
        type=int,
        help="step for evluation between [start, stop)")

    args = parser.parse_args()
    logger.info("args:{}".format(args))
    best_metrics = aggregate_all(args.work_dir, args.task_name, args.split, args.start, args.stop, args.step)
    logger.info("Best metrics: %s", json.dumps(best_metrics, indent=4))


if __name__ == "__main__":
    main()
