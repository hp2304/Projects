from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from tool.darknet2pytorch import Darknet
import argparse
import torch
import time
import cv2

"""hyper parameters"""
use_cuda = True
iou_threshold = 0.3
max_age = 20

def calculate_1d_iou(a1, a2, b1, b2):
    a1 = a1.view(-1, 1)
    a2 = a2.view(-1, 1)
    b1 = b1.view(1, -1)
    b2 = b2.view(1, -1)
    m = a1.shape[0]
    n = b1.shape[1]
    l_a = (a2 - a1)
    l_b = (b2 - b1)
    temp1 = torch.cat((a1.repeat(1, n).unsqueeze(0), b1.repeat(m, 1).unsqueeze(0)), axis=0)
    temp2 = torch.cat((a2.repeat(1, n).unsqueeze(0), b2.repeat(m, 1).unsqueeze(0)), axis=0)
    a_union_b = torch.max(temp2, axis = 0)[0] - torch.min(temp1, axis = 0)[0]
    a_inter_b = l_a + l_b - a_union_b
    a_inter_b[a_inter_b < 0] = 0
    return a_inter_b/a_union_b

def match(iou_mat, tracking_data, dets):
    new_id = torch.max(tracking_data[:, 7]).item() + 1
    max_vals, indices = torch.max(iou_mat, axis=1)
    dets_mask = max_vals > iou_threshold
    
    for i in range(max_vals.shape[0]):
        if dets_mask[i]:
            dets[i, 7] = tracking_data[indices[i], 7]
            tracking_data[indices[i], :7] = dets[i, :7]
            tracking_data[indices[i], 8] = 0
        else:
            dets[i, 7] = new_id
            new_id += 1

    new_dets = dets[~dets_mask]
    new_dets = torch.cat((new_dets, torch.zeros(new_dets.shape[0], 1)), axis=1)
    tracking_data = torch.cat((tracking_data, new_dets), axis=0)
    tracking_data[:, 8] += 1
    
    return tracking_data[tracking_data[:, 8] < max_age], dets

def detect_cv2_camera(cfgfile, weightfile, savename, outpath, classname, videosource = 0):
    import cv2
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(videosource)
    cap.set(3, 1280)
    cap.set(4, 720)

    if outpath:
        video_writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (1280, 720))

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)
    try:
        filter_classid = class_names.index(classname)
    except ValueError:
        print('Please provide valid classname. Refer to data/coco.names to see available class names.')
        exit()
    path = None
    cnt = 1
    is_first = True
    while True:
        ret, img = cap.read()
        if not ret:
            break
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        # boxes = torch.Tensor(do_detect(m, sized, 0.4, 0.6, use_cuda)[0])
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)[0]
        
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        boxes = boxes[boxes[:, 6] == filter_classid]
        if len(boxes) == 0:
            cv2.imshow('Yolov4', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if outpath:
                video_writer.write(img)
            if savename:
                path = savename + str(cnt) + '.jpg'
                cv2.imwrite(path, img)
            cnt += 1
            continue
        boxes = torch.cat((boxes, torch.arange(0, boxes.shape[0]).view(boxes.shape[0],1)), 1)
        

        if is_first:
            # Don't do matching
            tracking_data = torch.cat((boxes, torch.ones(boxes.shape[0], 1)), axis=1)
            is_first = False
        else:
            x_iou = calculate_1d_iou(boxes[:, 0], boxes[:, 2], tracking_data[:, 0], tracking_data[:, 2])
            y_iou = calculate_1d_iou(boxes[:, 1], boxes[:, 3], tracking_data[:, 1], tracking_data[:, 3])
            iou_mat = x_iou * y_iou
            tracking_data, boxes = match(iou_mat, tracking_data, boxes)

        if savename:
            path = savename + str(cnt) + '.jpg'
        result_img = plot_boxes_cv2(img, boxes, savename=path, class_names=class_names)

        cv2.imshow('Yolov4', result_img)
        if outpath:
            video_writer.write(result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cnt += 1

    cap.release()
    if outpath:
        video_writer.release()
    cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4-tiny.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/yolov4-tiny.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-videofile', type=str,
                        help='path of your video file.', dest='videofile')
    parser.add_argument('-savename', type=str,
                        default=None,
                        help='dirpath, where predictions of each frame will be stored', dest='savename')
    parser.add_argument('-outpath', type=str,
                        default=None,
                        help='dirpath, where output video will be stored', dest='outpath')
    parser.add_argument('-classname', type=str,
                        default='car',
                        help='Name of the class which you want to track(provide name from data/coco.names)', dest='classname')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.videofile:
        detect_cv2_camera(args.cfgfile, args.weightfile, args.savename, args.outpath, args.classname, args.videofile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile, args.savename, args.outpath, args.classname)
