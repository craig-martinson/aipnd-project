import argparse
from time import time
import json
import torch
import utility
import model_helper


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='flowers',
                        help='Path to the image files')

    return parser.parse_args()


def main():
    start_time = time()

    in_args = get_input_args()

    # Check for GPU
    use_gpu = torch.cuda.is_available()
    print("Running on {}".format("GPU" if use_gpu else "CPU"))

    # Loads a pretrained model
    model = model_helper.load_checkpoint('test.pth')
    # model = torch.load('full_save.pth')

    # Move tensors to GPU if available
    if use_gpu:
        model.cuda()

    # Get path to test image
    image_path = in_args.dir + '/test/28/image_05230.jpg'
    #image_path = test_dir + '/28/image_05230.jpg'
    #image_path = train_dir + '/1/image_06734.jpg'
    #image_path = train_dir + '/1/image_06735.jpg'

    # Load category mapping dictionary
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Prediction
    image_path = 'flowers/test/28/image_05230.jpg'
    print("Predication for: {}".format(image_path))
    probs, classes = model_helper.predict(image_path, model, use_gpu)
    print(probs)
    print(classes)

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    end_time = time()
    utility.print_elapsed_time(end_time - start_time)


# Call to main function to run the program
if __name__ == "__main__":
    main()
