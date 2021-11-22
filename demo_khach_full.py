import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.downloads import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.plots import plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
import datetime
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, savebus = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate, opt.savebus
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # print('out :',out)
    # print('source: ',source)
    # print('yolo_weights: ',yolo_weights)
    # print('deep_sort_weights: ',deep_sort_weights)
    # print('show_vid: ',show_vid)
    # print('save_vid: ',save_vid)
    # print('save_txt: ',save_txt)
    # print('imgsz: ',imgsz)
    # print('evaluate: ',evaluate)
    # print('savebus = ',savebus)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    # print('device: ',device)

    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # print('half: ',half)
    # print('device.type: ', device.type)

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # print('names: ', names)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    # print('txt_file_name: ',txt_file_name)
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    # print('txt_path: ',txt_path)
    i = 0
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # print(pred)

        k = 0
        for i in pred[0]:
            if int(i[5]) != 5:
                pred[0] = torch.cat([pred[0][0:k], pred[0][k+1:]])
                continue
            k = k + 1

        # print(pred)



        # print('pred: ',pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # print('i=',i)
            # print('det = ',det)
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string 384x640 
            # print(s)
            save_path = str(Path(out) / Path(p).name) #inference/output/sample_2s.mp4
            # print(save_path)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # print(s)

                xywhs = xyxy2xywh(det[:, 0:4])
                # print('xywhs = ',xywhs)
                confs = det[:, 4]
                # print('confs = ',confs)
                clss = det[:, 5]
                # print('clss = ',clss)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        #TODO: Minh
                        directory = str(id)
                        path_dir = savebus
                        if not os.path.exists(path_dir+directory):
	                        os.mkdir(os.path.join(path_dir, directory))


                        time_name = time.time()
                        name = path_dir+str(id)+'/'+str(id)+'_'+names[c]+'_'+str(round(time_name)*10)+'_'+str(datetime.datetime.now().timestamp()).split('.')[1]+'.jpg'
                        
                        x1, y1, x2, y2 = int(output[0]), int(output[1]), int(output[2]), int(output[3])
                        img_crop = im0[y1:y2, x1:x2]
                        cv2.imwrite(name, img_crop)
                        #------------------------
                        color = compute_color_for_id(id)
                        plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                               f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                        


            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

        # break


if __name__ == '__main__':
    for i in range(6,11):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5m.pt', help='model.pt path')
        parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str, default='data/xe_khach/'+str(i)+'.mp4', help='source')
        parser.add_argument('--output', type=str, default='inference/video'+str(i), help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
        parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
        parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
        # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--evaluate', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
        parser.add_argument('--savebus', type=str, default='./image_xe_khach_youtube/video'+str(i)+'/', help='save')  # output folder
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)

        with torch.no_grad():
            detect(args)
        # print('---------------------------')


#!python demo_track.py --yolo_weights /home/minh/Documents/minh/mqsolutions/object_tracking_yolov5/Yolov5_DeepSort_Pytorch/Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5m.pt --source data/xe_khach/1.mp4 --save-vid


