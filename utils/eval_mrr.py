import argparse
import csv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.metric import mean_reciprocal_rank

def main(csv_path):
    acc = 0
    num = 0
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                hum_id = row[0].split(".")[0]
                preds = []
                for col in row[1:]:
                    preds.append(str(col))

                print(hum_id, mean_reciprocal_rank(preds, str(hum_id)))
                acc += mean_reciprocal_rank(preds, str(hum_id))
                num += 1
            line_count += 1
        print(f'Processed {line_count} lines.')
    return acc / num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="path to predict csv")
    args = parser.parse_args()

    mrr = main(args.csv_path)
    print("-----------------------------")
    print(f"MRR: {mrr}")