import numpy as np
import colorsys 
from PIL import Image
import cv2
import tensorflow as tf

def load_coco_names(file_name):
    """Parses the label file with only class names,
      and returns a dictionary mapping the class IDs to class names.
    """
    names = {}
    with open(file_name) as f:
        for id_, name in enumerate(f):
            names[id_] = name
    return names

def load_labels(label_file):
    """Parses the label file, assuming that labels are separated with a newline
       in the file and returns the list of labels.
    """  
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [
        (x / len(class_names), 1., 1.) for x in range(len(class_names))
    ]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors

def letter_box_image(image: Image.Image, output_height: int, output_width: int, fill_value)-> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """

    height_ratio = float(output_height)/image.size[1]
    width_ratio = float(output_width)/image.size[0]
    fit_ratio = min(width_ratio, height_ratio)
    fit_height = int(image.size[1] * fit_ratio)
    fit_width = int(image.size[0] * fit_ratio)
    fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

    if isinstance(fill_value, int):
        fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

    to_return = np.tile(fill_value, (output_height, output_width, 1))
    pad_top = int(0.5 * (output_height - fit_height))
    pad_left = int(0.5 * (output_width - fit_width))
    to_return[pad_top:pad_top+fit_height, pad_left:pad_left+fit_width] = fit_image
    return to_return

def preprocess_image(image, model_image_size):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    # resized_image = letterbox_resize(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(image).astype('float32')
    image_data = image_data /255.0
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.
    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask
    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result

def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes
    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def draw_boxes(image,
               boxes,
               classes,
               scores,
               class_names,
               colors,
               show_score=True):
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0, 0, 0)
        else:
            color = colors[cls]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))

    return image

def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color,
                  cv2.FILLED)
    cv2.putText(
        image,
        text, (x + padding, y - text_height + padding),
        font,
        fontScale=font_scale,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA)

    return image

## converts the normalized positions  into integer positions
def darknet_to_original(width, height, box):

    xmax = int((box[0]*width) + (box[2] * width)/2.0)
    xmin = int((box[0]*width) - (box[2] * width)/2.0)
    ymax = int((box[1]*height) + (box[3] * height)/2.0)
    ymin = int((box[1]*height) - (box[3] * height)/2.0)
    return [xmin, xmax, ymin, ymax]

def infer_openvino_tensorflow(model_file, image_file , input_height, input_width, label_file, conf_threshold, iou_threshold):
    """Takes the tensorflow model and all other input parameters as arguments. 
       Runs inference with the object detection model and prints the predictions.
    """
    print("CREATE MODEL - BEGIN")

    # Load model and process input image
    model = tf.saved_model.load(model_file)
    print("CREATE MODEL - END")

    if label_file:
        labels = load_labels(label_file)
        colors = get_colors(labels)

    print("PREDICTION - BEGIN")
    
    #Preprocess Image
    image = Image.open(image_file)
    img = np.asarray(image)

    img_resized = letter_box_image(image, input_height, input_width, 128)
    preprocessed_image = preprocess_image(img_resized, (input_height, input_width))

    # pp_image = np.transpose(preprocessed_image, (0,3,1,2))
    img_resized = tf.convert_to_tensor(preprocessed_image)
    # Warmup iterations
    for _ in range(5):
        detected_boxes = model(img_resized)
        # detected_boxes = model(**{'images': img_resized})
        
    # Run
    import time
    elapsed = 0
    num_iterations = 10
    for _ in range(num_iterations):
        start = time.time()
        detected_boxes = model(img_resized)
        elapsed += time.time() - start
    result = non_max_suppression(detected_boxes[0].numpy(), 0.4,0.5)
    print('Average Inference time in ms: %f' % ((elapsed * 1000)/num_iterations))
    out_boxes = []
    out_classes = []
    out_scores = []
    for class_name, box_det in result.items():
        for box_score in box_det:
            score = box_score[1]
            box = box_score[0]
            box = darknet_to_original(640, 640,list(box))
            out_boxes.append(box)
            out_classes.append(class_name)
            out_scores.append(score)

    # img_bbox = draw_boxes(img, out_boxes, out_classes, out_scores,
    #                     labels, colors)
    # if output_dir:
    #     cv2.imwrite(os.path.join(output_dir, "detections.jpg"), img_bbox)
    # else:
    # cv2.imwrite("detections.jpg", img_bbox)
