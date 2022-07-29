import cv2
import time
import torch
import torchvision
import numpy as np
from typing import Tuple, Optional, Union
import glob
import os.path as osp
from typing import Callable

CLASS_LABELS = ['person']
COORDS = [[231,340],[491,320],[751,344],[938,571],[644,847],[298,856],[23,597],[115,450],[231,350]]

class DataStreamer(object):

    """Iterable DataStreamer class for generating numpy arr images
    Generates orig image and pre-processed image

    For loading data into detectors
    """

    def __init__(self, src_path: str, media_type: str = "image", preprocess_func: Callable = None):
        """Init DataStreamer Obj

        src_path : str
            path to a single image/video or path to directory containing images
        media_type : str
            inference media_type "image" or "video"
        preprocess_func : Callable function
            preprocessesing function applied to PIL images
        """
        if media_type not in {'video', 'image'}:
            raise NotImplementedError(
                f"{media_type} not supported in streamer. Use video or image")
        self.img_path_list = []
        self.vid_path_list = []
        self.idx = 0
        self.media_type = media_type
        self.preprocess_func = preprocess_func

        if media_type == "video":
            if osp.isfile(src_path):
                self.vid_path_list.append(src_path)
                self.vcap = cv2.VideoCapture(src_path)
            elif osp.isdir(src_path):
                raise NotImplementedError(
                    f"dir iteration supported for video media_type. {src_path} must be a video file")
        elif media_type == "image":
            if osp.isfile(src_path):
                self.img_path_list.append(src_path)
            elif osp.isdir(src_path):
                img_exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg']
                for ext in img_exts:
                    self.img_path_list.extend(
                        glob.glob(osp.join(src_path, ext)))

    def __iter__(self):
        return self

    def __next__(self):
        """Get next image or frame as numpy array

        """
        orig_img = None
        if self.media_type == 'image':
            if self.idx < len(self.img_path_list):
                orig_img = cv2.imread(self.img_path_list[self.idx])
                orig_img = orig_img[..., ::-1]
                self.idx += 1
        elif self.media_type == 'video':
            if self.idx < len(self.vid_path_list):
                ret, frame = self.vcap.read()
                if ret:
                    orig_img = frame[..., ::-1]
                else:
                    self.idx += 1
        if orig_img is not None:
            proc_img = None
            if self.preprocess_func is not None:
                proc_img = self.preprocess_func(orig_img)
                proc_img = np.expand_dims(proc_img, axis=0)
            return np.array(orig_img), proc_img
        raise StopIteration


def apply_mask(img: cv2.image, coords: list = COORDS) -> cv2.image:
    """
    Imports the image --> creates mask --> applies mask --> returns image
    list of coord: [[coord1 width, coord1 height],[],....]
    """
    mask = np.zeros(img.shape[:2], dtype=np.int8)
    shape = np.array(coords, np.int32)
    cv2.fillConvexPoly(mask, shape, 1)
    # apply our mask 
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


def preprocess_image(
    cv2_img: np.ndarray,
    in_size: Tuple[int, int] = (960, 960),
    coords: list = COORDS
) -> np.ndarray:
    """preprocesses cv2 image and returns a norm np.ndarray

        cv2_img = cv2 image
        in_size: in_width, in_height
    """
    resized = center_crop_img(cv2_img, in_size)
    masked = apply_mask(resized, coords)
    img_in = np.transpose(masked, (2, 0, 1)).astype('float32')  # HWC -> CHW
    img_in /= 255.0
    return img_in


def save_output(
    detections,
    image_src: np.ndarray,
    save_path: str,
    threshold: float,
    model_in_HW: Tuple[int, int],
    line_thickness: Optional[int] = None,
    text_bg_alpha: float = 0.0
) -> None:
    image_src = cv2.cvtColor(image_src, cv2.COLOR_RGB2BGR)
    labels = detections[..., -1].numpy()
    boxs = detections[..., :4].numpy()
    confs = detections[..., 4].numpy()

    if isinstance(image_src, str):
        image_src = cv2.imread(image_src)
    elif isinstance(image_src, np.ndarray):
        image_src = image_src

    mh, mw = model_in_HW
    h, w = image_src.shape[:2]
    boxs[:, :] = scale_coords((mh, mw), boxs[:, :], (h, w)).round()
    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        if confs[i] >= threshold:
            x1, y1, x2, y2 = map(int, box)
            np.random.seed(int(labels[i]) + 2020)
            color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
            cv2.rectangle(image_src, (x1, y1), (x2, y2), color, thickness=max(
                int((w + h) / 600), 1), lineType=cv2.LINE_AA)
            label = '%s %.2f' % (CLASS_LABELS[int(labels[i])], confs[i])
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=1)[0]
            c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
            if text_bg_alpha == 0.0:
                cv2.rectangle(image_src, (x1 - 1, y1), c2,
                              color, cv2.FILLED, cv2.LINE_AA)
            else:
                # Transparent text background
                alphaReserve = text_bg_alpha  # 0: opaque 1: transparent
                BChannel, GChannel, RChannel = color
                xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
                xMax, yMax = int(x1 + t_size[0]), int(y1)
                image_src[yMin:yMax, xMin:xMax, 0] = image_src[yMin:yMax,
                                                               xMin:xMax, 0] * alphaReserve + BChannel * (1 - alphaReserve)
                image_src[yMin:yMax, xMin:xMax, 1] = image_src[yMin:yMax,
                                                               xMin:xMax, 1] * alphaReserve + GChannel * (1 - alphaReserve)
                image_src[yMin:yMax, xMin:xMax, 2] = image_src[yMin:yMax,
                                                               xMin:xMax, 2] * alphaReserve + RChannel * (1 - alphaReserve)
            cv2.putText(image_src, label, (x1 + 3, y1 - 4), 0, tl / 3, [255, 255, 255],
                        thickness=1, lineType=cv2.LINE_AA)
            print("bbox:", box, "conf:", confs[i],
                  "class:", CLASS_LABELS[int(labels[i])])
    cv2.imwrite(save_path, image_src)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.55,
    iou_thres: float = 0.6,
    classes: Optional[torch.Tensor] = None,
    # agnostic: bool = False,
    # multi_label: bool = False,
    # labels: Tuple[str] = ()
) -> torch.Tensor:
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) maximum and minimum box width and height
    max_wh = 4096  # min_wh = 2
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    # multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # if multi_label:
        #     i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        #     x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        # else:  # best class only
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[
            conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def pad_resize_image(
    cv2_img: np.ndarray,
    new_size: Tuple[int, int] = (960, 960),
    color: Tuple[int, int, int] = (125, 125, 125)
) -> np.ndarray:
    """Resize and pad image with color if necessary, maintaining orig scale

    args:
        cv2_img: numpy.ndarray = cv2 image
        new_size: tuple(int, int) = (width, height)
        color: tuple(int, int, int) = (B, G, R)
    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    # rescale down
    scale = min(new_w / in_w, new_h / in_h)
    # get new sacled widths and heights
    scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
    resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))
    # calculate deltas for padding
    d_w = max(new_w - scale_new_w, 0)
    d_h = max(new_h - scale_new_h, 0)
    # center image with padding on top/bottom or left/right
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    pad_resized_img = cv2.copyMakeBorder(resized_img,
                                         top, bottom, left, right,
                                         cv2.BORDER_CONSTANT,
                                         value=color)
    return pad_resized_img

def center_crop_img(
    cv2_img: np.ndarray,
    new_size: Tuple[int, int] = (960, 960)) -> np.ndarray:
    """
    Given a new size and cv2 image, crops into the center of the image
    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    h_diff = int((in_h - new_h)/2)
    w_diff = int((in_w-new_w)/2)
    cropped_img = cv2_img[h_diff:h_diff + new_h, w_diff:w_diff + new_w]
    return cropped_img

def clip_coords(
    boxes: Union[torch.Tensor, np.ndarray],
    img_shape: Tuple[int, int]
) -> None:
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    else:  # np.array
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def scale_coords(img1_shape: Tuple[int, int], coords: np.ndarray, img0_shape: Tuple[int, int], ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
