import argparse
from time import time
import json
import torch
import utility
import model_helper


def get_input_args():
    parser = argparse.ArgumentParser()

    # Add positional arguments
    parser.add_argument('input', type=str,
                        help='Input image')

    parser.add_argument('checkpoint', type=str,
                        help='Model checkpoint file to use for prediction')

    # Add optional arguments
    parser.add_argument('--top_k', type=int, default=3,
                        help='Return top k most likely classes')

    parser.add_argument('--category_names', type=str,
                        help='Mapping file used to map categories to real names')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU for prediction')
    parser.set_defaults(gpu=False)

    return parser.parse_args()


def get_title(label, cat_to_name):
    try:
        return cat_to_name[label]
    except KeyError:
        return "unknown label"


def main():
    start_time = time()

    in_args = get_input_args()

    # Check for GPU
    use_gpu = torch.cuda.is_available() and in_args.gpu

    # Print parameter information
    print("Predicting on {} using {}".format(
        "GPU" if use_gpu else "CPU", in_args.checkpoint))

    # Loads a pretrained model
    model = model_helper.load_checkpoint(in_args.checkpoint)

    # Move tensors to GPU if available
    if use_gpu:
        model.cuda()

    # Load category mapping dictionary
    use_mapping_file = False

    if in_args.category_names:
        with open(in_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            use_mapping_file = True

    # Get prediction
    probs, classes = model_helper.predict(
        in_args.input, model, use_gpu, in_args.top_k)

    # Print results
    print("\nTop {} Classes predicted for '{}':".format(
        len(classes), in_args.input))

    if use_mapping_file:
        print("\n{:<30} {}".format("Flower", "Probability"))
        print("{:<30} {}".format("------", "-----------"))
    else:
        print("\n{:<10} {}".format("Class", "Probability"))
        print("{:<10} {}".format("------", "-----------"))

    for i in range(0, len(classes)):
        if use_mapping_file:
            print("{:<30} {:.2f}".format(
                get_title(classes[i], cat_to_name), probs[i]))
        else:
            print("{:<10} {:.2f}".format(classes[i], probs[i]))

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    end_time = time()
    utility.print_elapsed_time(end_time - start_time)


# Call to main function to run the program
if __name__ == "__main__":
    main()
