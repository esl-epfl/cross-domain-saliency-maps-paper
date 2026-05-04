import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process experiment results and aggregate metrics.')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The name or path of the model to use.'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the dataset.'
    )

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='',
        help='A name for the experiment. Default is "_final".'
    )

    parser.add_argument(
        '--top_value',
        type=int,
        default=100,
        help='An integer representing the top value parameter. Default is 100.'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    model = args.model
    data = args.data
    experiment_name = args.experiment_name
    top_value = args.top_value

    # Define the base file name pattern
    file_pattern = f"./results/{data}/{model}_{top_value}{experiment_name}.csv"

    results = pd.read_csv(file_pattern)

    print(f"== {data} ==")
    print('Average : ', results['mean_top50'].iloc[0], ' (+- ', results['std_error_top50'].iloc[0], ' )')
    print('Zero : ', results['mean_top50'].iloc[1], ' (+- ', results['std_error_top50'].iloc[1], ' )')

if __name__ == '__main__':
    main()