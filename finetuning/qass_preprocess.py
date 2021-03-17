import argparse
import json
import os
from glob import glob
from tqdm import tqdm


MASK_TOKEN = "[MASK]"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    return args


def insert_mask_to_qa_end(data_point):
    transformed_data_point = data_point.copy()
    qas = []
    for qa in transformed_data_point["qas"]:
        qa = qa.copy()
        qa["question"] += f" {MASK_TOKEN}."
        qas.append(qa)
    transformed_data_point["qas"] = qas
    return transformed_data_point


def write_mrqa_single_path(data, path, header):
    output_path = os.path.splitext(path)[0] + "_qass.jsonl"
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write(header)
            for dp in data:
                f.write(json.dumps(insert_mask_to_qa_end(dp)) + '\n')


def read_mrqa(path):
    data = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            data_point = json.loads(line)
            data.append(data_point)
    return data, header


def main():
    args = get_args()
    paths = glob(args.path)
    for path in tqdm(paths):
        data, header = read_mrqa(path)
        write_mrqa_single_path(data, path, header)


if __name__ == '__main__':
    main()
