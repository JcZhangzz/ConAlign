import argparse

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--dataset', type=str, default='yahooR3', help='Choose from {yahooR3, coat, simulation}')
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    parser.add_argument('--test_only', action='store_true', help='Only run evaluation using saved model weights.')
    return parser.parse_args()
