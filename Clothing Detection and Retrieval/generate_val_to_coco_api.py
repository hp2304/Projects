import argparse
import pickle

from torchvision import transforms

from coco_utils import get_coco_api_from_dataset
from dataset_helpers import ClothesDataset

'''
epoch 3:

Test: Total time: 0:21:19 (0.0796 s / it)
Averaged stats: model_time: 0.0584 (0.0602)  evaluator_time: 0.0129 (0.0179)
Accumulating evaluation results...
DONE (t=5.51s).
Accumulating evaluation results...
DONE (t=6.71s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.679
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.963
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.789
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.753
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.753
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.755
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.723
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.962
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.837
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.726
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.778
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779

'''
def main(args):
    val_transform = transforms.Compose([])

    dataset = ClothesDataset(args.dataset_path, transform = val_transform, is_train = False)

    coco_ds = get_coco_api_from_dataset(dataset)
    with open(args.out_fpath, 'wb') as fp:
        pickle.dump(coco_ds, fp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--dataset_path', 
                        type=str,
                        required=True,
                        dest='dataset_path',
                        help='Directory containing images')

    
    parser.add_argument('-o', '--out_fpath', 
                        type=str,
                        required=True,
                        dest='out_fpath',
                        help='Output pickle filepath')

    args = parser.parse_args()
    main(args)