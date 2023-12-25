import argparse
from PIL import Image
import os
import json

def main(args):
    size_metadata = {}
    for filename in os.listdir(args.img_dir_path):
        fp = os.path.join(args.img_dir_path, filename)
        w, h = Image.open(fp).convert('RGB').size
        size_metadata[filename] = {'width': w, 'height': h}

    with open(args.out_fpath, 'w') as fp:
        json.dump(size_metadata, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--img_dir', 
                        type=str,
                        required=True,
                        dest='img_dir_path',
                        help='Directory containing images')
    parser.add_argument('-o', '--out_fpath', 
                        type=str,
                        required=True,
                        dest='out_fpath',
                        help='Output JSON filepath')

    args = parser.parse_args()
    main(args)