import argparse
from builtins import str
from collections import defaultdict
import json
from pathlib import Path
import os, sys
import numpy as np
from numpy import mean, std
from itertools import product
from run_method import run_method
from datetime import datetime
from utils import Benchmark, verify_and_load_json_dataset
from pdb import set_trace as stop
# 设置不同的数据路径
ROOT_DIR = Path(__file__).resolve().parent.parent
# 数据来源路径
DATA_DIR = ROOT_DIR / "ATEdata"
# 模型的参数路径
EVAL_DIR = ROOT_DIR / "eval"
TMP_HPARAMS_PATH = "./absa_hparams.json"
METRICS = ("precision", "recall", "f1")
bench = Benchmark()
track = bench.track
# SEEDS = (1, 2, 3, 4, 5)
SEEDS = (12, 27, 42, 20, 33)


def split_sizes():
    for dataset in 'rest', 'device', 'lap':
        ds_path = f"{DATA_DIR}/{dataset}"
        print(f"\n\n{dataset}")
        dataset = verify_and_load_json_dataset(f"{ds_path}.json")["train"]
        # Split data pool to d_test (1/3) and d_unlabeled (2/3)
        splits = dataset.train_test_split(seed=12, test_size=1/4)
        d_test = splits["test"]
        d_unlabeled = splits["train"]
        print(f"test len: {len(d_test)}")
        print(f"unlabeled len: {len(d_unlabeled)}")
        d_unlabeled_pos = d_unlabeled.filter(lambda ex: 'B-ASP' in ex['tags'])
        print(f"max positive train pos len: {len(d_unlabeled_pos)}")

def reconstruct_report(results_a_json, results_b_json, output_dir, sample_size):
    hparam_space = load_hparam_space()
    with open(results_a_json) as f_a:
        res_a = json.load(f_a)
    with open(results_b_json) as f_b:
        res_b = json.load(f_b)
    full_res = res_a + res_b
    assert len(full_res) == len(hparam_space)
    f1_means = [s["f1_mean"] for s in full_res]
    best_hparam_index = np.argmax(f1_means)
    best_hparams = hparam_space[best_hparam_index]
    best_hparams_json = json.dumps(best_hparams, indent=4)
    with open(output_dir / f"ex={sample_size}.json", "w") as f:
        f.write(best_hparams_json)
    return best_hparams

def load_hparam_space(path, start_from=0):
    # The input is a path to a json file containing a dict of hparam space:
    # the keys are hparam names (str) and the values are a single hparam value or a list of possible values.
    # Returns list of dict, each dict contains 2 keys: "tune" (tunable hparams) and "fixed" (untunable hparams)
    with open(EVAL_DIR / "hparam_space" / path) as f:
        hparams_dict = json.load(f)
    tune, fixed = {}, {}
    for k, v in hparams_dict.items():
        (tune if isinstance(v, list) and len(v) > 1 else fixed)[k] = v
    hparam_space = [{"tune": dict(comb), "fixed": dict(fixed)} for comb in \
        product(*[[(k, v) for v in vs] for k, vs in tune.items()])]
    return hparam_space[start_from:]

def write_results_summary(results, path):
    metrics = 'precision', 'recall', 'f1'
    with open(path, 'w') as f:
        headers = ["#ex"]
        for m in metrics:
            headers.append(m[0].upper())
            headers.append(f"{m[0].upper()}_std")
        f.write('\t'.join(headers) + '\n')
        for num_ex, values in results.items():
            f.write(f"{num_ex}\t")
            for m in metrics:
                mean_key, std_key = f"{m}_mean", f"{m}_std"
                f.write(f"{str(round(values[mean_key], 4))}\t")
                f.write(f"{str(round(values[std_key], 4))}\t")
            f.write('\n')


def init_output_dir(subdir, data):
    # 保存输出结果的路径
    output_dir = EVAL_DIR / subdir / \
        (datetime.now().strftime("%d.%m-%H:%M:%S") + "_" + data)
    latest_dir = EVAL_DIR / subdir / f"{data}_latest"
    # 符号链接，通过output_dir指向latest_dir，最后返回output_dir
    os.symlink(latest_dir, output_dir)
    return output_dir


# 加载数据集的函数
def load_data(dataset, pos_ex_only=False):
    # 数据集的地址
    ds_path = f"{DATA_DIR}/{dataset}"
    # label数据集
    d_label = verify_and_load_json_dataset(f"{ds_path}_unlabeled.json")["train"]
    # 抽取包含属性词的句子
    if pos_ex_only:
        d_label = d_label.filter(lambda ex: 'B-ASP' in ex['tags'])
    # test数据集的量
    d_test = verify_and_load_json_dataset(f"{ds_path}_test.json")["train"]
    # unlabel数据集的量
    d_unlabeled = verify_and_load_json_dataset(f"{ds_path}_unlabeled.json")["train"]
    return d_label, d_test, d_unlabeled


def run_method_with_hparams(dataset: str, hparams: dict=None, seed=42, train=None, dev=None, test=None, unlabeled=None):
    hparams_flat = {**hparams["tune"], **hparams["fixed"]} if "tune" in hparams else hparams
    with open(TMP_HPARAMS_PATH, "w") as f:
        json.dump(hparams_flat, f, indent=4)
    # 有train、test而没有dev数据集
    metrics = run_method(dataset=dataset, hparams_path=TMP_HPARAMS_PATH, seed=seed,\
        train=train, dev=dev, test=test, unlabeled=unlabeled)
    return metrics


class TuneFewShot:
    def __init__(self, data: str=None, hparam_space: str=None, num_splits: int=5,
                seed=42, max_train_labels=1000, sample_sizes: tuple=(32,), start_from=0) -> None:
        self.seed = seed
        self.data = data
        self.max_train_labels = max_train_labels
        self.num_splits = num_splits
        self.sample_sizes = sample_sizes
        # 传入label、test、unlabeled等数据集
        self.d_label, self.d_test, self.d_unlabeled = self.split_into_label_and_test(data)
        self.hparam_space = load_hparam_space(hparam_space, start_from)
        self.output_dir = init_output_dir("best_hparams", data)


    # 在初始化的时候就调用该函数
    def split_into_label_and_test(self, dataset: str, max_num_label: int=1000):
        """
        Randomly split a labeled dataset with N samples into 2 sets:
        d_label with `max_num_label` samples, and d_test with N - `max_num_label` samples.
        Write the sets to the data folder.
        """
        # 数据的位置信息
        ds_path = f"{DATA_DIR}/{dataset}"
        # 标签路径（训练集？？）、测试集路径
        label_path, test_path = f"{ds_path}_label.json", f"{ds_path}_test.json"
        unlabeled_path = f"{ds_path}_unlabeled.json"
        if not os.path.exists(label_path) or not os.path.exists(test_path):
            dataset = verify_and_load_json_dataset(f"{ds_path}.json")["train"]
            # Split data pool to d_test (1/3) and d_unlabeled (2/3)
            splits = dataset.train_test_split(seed=self.seed, test_size=1/3)
            d_test = splits["test"]
            d_unlabeled = splits["train"]
            # d_label consits of first 1000 samples from d_unlabeled
            d_label = d_unlabeled.select(range(max_num_label))
            d_test.to_json(test_path)
            d_label.to_json(label_path)
            d_unlabeled.to_json(unlabeled_path)
        else:
            d_label = verify_and_load_json_dataset(label_path)["train"]
            d_test = verify_and_load_json_dataset(test_path)["train"]
            d_unlabeled = verify_and_load_json_dataset(unlabeled_path)["train"]
        return d_unlabeled, d_test, d_unlabeled


    # 对所有数据集得到最好的在验证集上的性能
    def tune_all_samples_sizes(self):
        report = {}
        for sample_size in self.sample_sizes:
            print(f"\n\n Tuning with sample size = {sample_size}... \n\n")
            result = self.tune(sample_size)
            report[sample_size] = result["report"]
            # Print best hparams for this sample size, and write to file
            best_hparams_json = json.dumps(result["best_hparams"], indent=4)
            with open(self.output_dir / f"ex={sample_size}.json", "w") as f:
                f.write(best_hparams_json)
        # Print summary report and write to file
        report_json = json.dumps(report, indent=4)
        print(f"\n\nTuning Report: \n{report_json}")
        with open(self.output_dir / f"report.json", "w") as f:
           f.write(report_json)

    # 通过对label数据集进行划分，然后得到在验证集上的最优性能
    def tune(self, sample_size: int):
        print(f"\n\n Tuning with sample size {sample_size}... \n\n")
        # 从d_label中选择sample_size个数据给d_sample_size
        d_sample_size = self.d_label.select(range(sample_size))
        multi_splits = self.create_multi_splits(d_sample_size)
        report = []
        for i, h_set in enumerate(self.hparam_space):
            print(f"\n\n********** Hyperparameters: **********")
            print(json.dumps(h_set, indent=4))
            scores = defaultdict(list)
            with track(f"hparam set: \n{i + 1} out of {len(self.hparam_space)} hparam_sets"):
                # 使用5*16的数据集来进行调整参数
                for i, splits in enumerate(multi_splits):
                    with track(f"Running multi-split {i + 1}/{len(multi_splits)}"):
                        # unlabeled总共有2565个数据
                        result = run_method_with_hparams(dataset=self.data, hparams=h_set,
                                    train=splits["train"],
                                    dev=splits["dev"],
                                    unlabeled=self.d_unlabeled)
                        for metric in METRICS:
                            scores[metric].append(result["dev"][f"dev_{metric}"])
            agg_result = {f"{m}_{f.__name__}": f(scores[m]) for f in (mean, std) for m in METRICS}
            agg_result["hparams"] = h_set["tune"]
            report.append(agg_result)
            # 通过运行方法M在训练集，以及评估在验证集上，报道最优的在验证集上的性能
            with open(self.output_dir / "intermediate_results.json", "w") as f:
                json.dump(report, f, indent=4)
        best_hparam_index = np.argmax([s["f1_mean"] for s in report])
        best_hparams = self.hparam_space[best_hparam_index]
        return {"best_hparams": best_hparams, "report": report}

    # num_splits=5,d_label中选择数据集组成multi_splits
    # 随机将label标签的数据集划分为训练集和验证集
    def create_multi_splits(self, d_sample_size):
        multi_splits = []
        for k in range(self.num_splits):
            splits = d_sample_size.train_test_split(seed=k, test_size=0.5)
            multi_splits.append({"train": splits["train"], "dev": splits["test"]})
        return multi_splits


class TuneBaseline:
    def __init__(self, data: str=None, hparam_space: str=None, num_splits: int=5,
                seed=42, max_train_labels=1000, sample_sizes: tuple=(32,), start_from=0) -> None:
        self.seed = seed
        self.data = data
        self.max_train_labels = max_train_labels
        self.num_splits = num_splits
        self.sample_sizes = sample_sizes
        self.d_label, self.d_test, _ = self.split_into_label_and_test(data)
        self.hparam_space = load_hparam_space(hparam_space, start_from)
        self.output_dir = init_output_dir("best_hparams_base", data)

    def split_into_label_and_test(self, dataset: str, max_num_label: int=1000):
        """
        Randomly split a labeled dataset with N samples into 2 sets:
        d_label with `max_num_label` samples, and d_test with N - `max_num_label` samples.
        Write the sets to the data folder.
        """
        ds_path = f"{DATA_DIR}/{dataset}"
        label_path, test_path = f"{ds_path}_label.json", f"{ds_path}_test.json"
        unlabeled_path = f"{ds_path}_unlabeled.json"
        if not os.path.exists(label_path) or not os.path.exists(test_path):
            dataset = verify_and_load_json_dataset(f"{ds_path}.json")["train"]
            # Split data pool to d_test (1/3) and d_unlabeled (2/3)
            splits = dataset.train_test_split(seed=self.seed, test_size=1/3)
            d_test = splits["test"]
            d_unlabeled = splits["train"]
            # d_label consits of first 1000 samples from d_unlabeled
            d_label = d_unlabeled.select(range(max_num_label))
            d_test.to_json(test_path)
            d_label.to_json(label_path)
            d_unlabeled.to_json(unlabeled_path)
        else:
            d_label = verify_and_load_json_dataset(label_path)["train"]
            d_test = verify_and_load_json_dataset(test_path)["train"]
            d_unlabeled = verify_and_load_json_dataset(unlabeled_path)["train"]
        return d_label, d_test, d_unlabeled

    def tune_all_samples_sizes(self):
        report = {}
        for sample_size in self.sample_sizes:
            print(f"\n\n Tuning with sample size = {sample_size}... \n\n")
            result = self.tune(sample_size)
            report[sample_size] = result["report"]

            # Print best hparams for this sample size, and write to file
            best_hparams_json = json.dumps(result["best_hparams"], indent=4)
            with open(self.output_dir / f"ex={sample_size}.json", "w") as f:
                f.write(best_hparams_json)
        # Print summary report and write to file
        report_json = json.dumps(report, indent=4)
        print(f"\n\nTuning Report: \n{report_json}")
        with open(self.output_dir / f"report.json", "w") as f:
           f.write(report_json)


    def tune(self, sample_size: int):
        print(f"\n\n Tuning with sample size {sample_size}... \n\n")
        d_sample_size = self.d_label.select(range(sample_size))
        multi_splits = self.create_multi_splits(d_sample_size)
        report = []
        for i, h_set in enumerate(self.hparam_space):
            print(f"\n\n********** Hyperparameters: **********")
            print(json.dumps(h_set, indent=4))
            scores = defaultdict(list)

            with track(f"hparam set: \n{i + 1} out of {len(self.hparam_space)} hparam_sets"):
                for i, splits in enumerate(multi_splits):
                    with track(f"Running multi-split {i + 1}/{len(multi_splits)}"):
                        result = run_method_with_hparams(dataset=self.data, hparams=h_set,
                                            train=splits["train"],
                                            dev=splits["dev"])
                        for metric in METRICS:
                            scores[metric].append(result["dev"][f"dev_{metric}"])
            agg_result = {f"{m}_{f.__name__}": f(scores[m]) for f in (mean, std) for m in METRICS}
            agg_result["hparams"] = h_set["tune"]
            report.append(agg_result)
            with open(self.output_dir / "intermediate_results.json", "w") as f:
                json.dump(report, f, indent=4)
        best_hparam_index = np.argmax([s["f1_mean"] for s in report])
        best_hparams = self.hparam_space[best_hparam_index]
        return {"best_hparams": best_hparams, "report": report}


    def create_multi_splits(self, d_sample_size):
        multi_splits = []
        for k in range(self.num_splits):
            splits = d_sample_size.train_test_split(seed=k, test_size=0.5)
            multi_splits.append({"train": splits["train"], "dev": splits["test"]})
        return multi_splits


class TestFewShot:
    def __init__(self, data: str=None, hparams: str=None,
                sample_sizes: tuple=(8, 16, 32, 48, 64, 80, 100, 200, 1000), seeds=(12, 27, 6)) -> None:
        self.sample_sizes = sample_sizes
        self.output_dir = init_output_dir("test_results", data)
        self.seeds = seeds
        self.data = data
        open(self.output_dir / f'seeds_{self.seeds}', 'w')
        with open(EVAL_DIR /"best_hparams" / hparams) as f:
            self.hparams = json.load(f)
        with open(self.output_dir / 'hparams.json', 'w') as f:
            json.dump(self.hparams, f, indent=4)
        # 导入label、test、unlabeled数据集 
        self.d_label, self.d_test, self.d_unlabeled = load_data(data, pos_ex_only=False)


    def test_all_samples_sizes(self):
        results = {}
        # for sample_size in self.sample_sizes:
        sample_size = len(self.d_label)
        # for sample_size in self.sample_sizes:
        print(f"\n\n Testing with sample size = {sample_size}... \n\n")
        agg_scores = self.test(sample_size)
        results[sample_size] = agg_scores
        # DEBUG: Write averages
        with open(self.output_dir / f"intermediate_avg.json", "w") as f:
            json.dump(results, f, indent=4)
        # Print summary report and write to file
        write_results_summary(results, self.output_dir / f"test_results.txt")
        

    def test(self, sample_size: tuple):
        scores = defaultdict(list)
        seed_f1 = {}
        for seed in self.seeds:
            print(f"\n\n Testing with seed = {seed}... \n\n")
            d_sample_size = self.d_label.shuffle(seed=seed).select(range(sample_size))
            result = run_method_with_hparams(dataset=self.data, seed=seed,
                                hparams=self.hparams,
                                train=d_sample_size,
                                test=self.d_test,
                                unlabeled=self.d_unlabeled)
            # DEBUG: Write seed metrics
            seed_f1[seed] = result["test"]
            with open(self.output_dir / f"intermediate_ex={sample_size}.json", "w") as f:
                json.dump(seed_f1, f, indent=4)
            for metric in METRICS:
                scores[metric].append(result["test"][f"test_{metric}"])
        agg_scores = {f"{m}_{f.__name__}": f(scores[m]) for f in (mean, std) for m in METRICS}
        return agg_scores


class TestBaseline:
    def __init__(self, data: str=None, hparams: str=None,
                sample_sizes: tuple=(8, 16, 32, 48, 64, 80, 100, 200, 1000), seeds=(12, 27, 6)) -> None:
        self.d_label, self.d_test, _ = load_data(data)
        self.sample_sizes = sample_sizes
        self.output_dir = init_output_dir("baseline_results", data)
        self.seeds = seeds
        self.data = data
        open(self.output_dir / f'seeds_{self.seeds}', 'w')
        with open(EVAL_DIR / "best_hparams_base" / hparams) as f:
            self.hparams = json.load(f)
        with open(self.output_dir / 'hparams.json', 'w') as f:
            json.dump(self.hparams, f, indent=4)

    def test_all_samples_sizes(self):
        results = {}
        for sample_size in self.sample_sizes:
            print(f"\n\n Testing baseline with sample size = {sample_size}... \n\n")
            agg_scores = self.test(sample_size)
            results[sample_size] = agg_scores
            # DEBUG: Write averages
            with open(self.output_dir / f"intermediate_avg.json", "w") as f:
                json.dump(results, f, indent=4)
        # Print summary report and write to file
        write_results_summary(results, self.output_dir / f"test_results.txt")

    def test(self, sample_size: int):
        scores = defaultdict(list)
        seed_f1 = {}
        for seed in self.seeds:
            print(f"\n\n Testing baseline with seed = {seed}... \n\n")
            d_sample_size = self.d_label.shuffle(seed=seed).select(range(sample_size))
            result = run_method_with_hparams(dataset=self.data, seed=seed,
                                hparams=self.hparams,
                                train=d_sample_size,
                                test=self.d_test)
            # DEBUG: Write seed metrics
            seed_f1[seed] = result["test"]
            with open(self.output_dir / f"intermediate_ex={sample_size}.json", "w") as f:
                json.dump(seed_f1, f, indent=4)
            for metric in METRICS:
                scores[metric].append(result["test"][f"test_{metric}"])
        agg_scores = {f"{m}_{f.__name__}": f(scores[m]) for f in (mean, std) for m in METRICS} 
        return agg_scores
 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("res14", "res15", "res16", "lap14"), default="res14")
    parser.add_argument("--task", choices=("tune", "test", "tune_base", "test_base"), default="test")
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    return args

    
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.task == "tune":
    #  调整模型参数
        print('----------------------------------------------Tuning Time-------------------------------------------------') 
        TuneFewShot(data=args.dataset, hparam_space="model.json", num_splits=5,
        seed=27, max_train_labels=1000, start_from=0)\
            .tune_all_samples_sizes()
    #  训练和评估模型使用调整完之后的参数
    if args.task == "test":
        print('----------------------------------------------Time for testing-------------------------------------------------')
        TestFewShot(data=args.dataset, hparams=f"{args.dataset}_latest/ex=32.json", seeds=SEEDS)\
            .test_all_samples_sizes()

if __name__ == "__main__":
    main()