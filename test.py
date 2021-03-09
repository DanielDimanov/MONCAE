import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
args = parser.parse_args()
print(args.dataset)