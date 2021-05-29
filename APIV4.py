from flask_cors import CORS
import json
import io
import base64
from waitress import serve
from eventlet import wsgi
import eventlet
from flask import Flask, request, Response
from flask_cors import cross_origin
import numpy as np
import cv2
from PIL import Image, ImageOps, ExifTags
from PIL.ExifTags import TAGS
from tensorflow.python.saved_model import tag_constants
import forV4.core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import Preprocessing_img as pre

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================


_FLAGS = {
    'framework': 'tf',
    'weights': './forV4/checkpoints/yolov4-416',
    'size': 416,
    'tiny': False,
    'model': 'yolov4',
    'images': '',
    'output': '',
    'iou': 0.45,
    'score': 0.4,
    'dont_show': False
}
saved_model_loaded = tf.saved_model.load(
    _FLAGS['weights'], tags=[tag_constants.SERVING])


def main_pre(FLS, IM):
    original_image = IM
    input_size = FLS['size']
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLS['iou'],
        score_threshold=FLS['score']
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                 valid_detections.numpy()]
    image, listImageCrop = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    # if not FLAGS.dont_show:
    #     image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLS['output'] + 'detection' + '.png', image)
    return image, listImageCrop


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def predictOne(img):
    npIMG = np.array(img)

    print(npIMG.shape)
    # cv2.imshow('test2', npIMG)
    # cv2.waitKey(0)
    print('gamma before correct', np.mean(npIMG, axis=(0, 1)))
    npIMG = pre.gamma_correction(npIMG, gamma=1.2)
    print('gamma after correct', np.mean(npIMG, axis=(0, 1)))

    imageToCrop = npIMG
    FRsult = list()
    result, listImagesCrop = main_pre(_FLAGS, npIMG)
    if (len(listImagesCrop) > 0):
        for ObjTarget in listImagesCrop:
            thisKey = ObjTarget.keys()
            label = list(thisKey)[0]
            score = list(thisKey)[1]
            if (ObjTarget[score] > 0.4):
                print('asdasdasd', label)
                startX = 0
                endX = 0
                startY = 0
                endY = 0
                print('adasdsadasdasd', label)
                if (ObjTarget[label]['startx'] >= 0):
                    startX = int(ObjTarget[label]['startx'])
                if (ObjTarget[label]['endx'] >= 0):
                    endX = int(ObjTarget[label]['endx'])

                if (ObjTarget[label]['starty'] >= 0):
                    startY = int(ObjTarget[label]['starty'])
                if (ObjTarget[label]['endy'] >= 0):
                    endY = int(ObjTarget[label]['endy'])
                print('-----------------------', startX, endX, startY, endY)
                image = imageToCrop[
                        startY:endY, startX:endX
                        ]
                image = pre.text_filter(image)

                image = Image.fromarray(image)
                image = image_to_byte_array(image)
                image = base64.b64encode(image).decode()
                FRsult.append({
                    label: image
                })
        result = Image.fromarray(result)
        result = image_to_byte_array(result)
        FRsult.append({
            'imgWithBox': base64.b64encode(result).decode()
        })
        final_new_data = json.dumps(
            {'files': FRsult}, sort_keys=True, indent=4, separators=(',', ': ')
        )
        return final_new_data


@app.route('/api/predict', methods=['POST'])
def predictByV4():
    print('recieve a request')
    data = json.loads(request.form['img'])['img']
    if (len(data) == 1):
        base64_form = data[0]
        base64_data = base64_form.split('base64,')[1]
        byte_img = io.BytesIO(base64.b64decode(base64_data))
        # img = request.files["image"].read()
        img = Image.open(byte_img)
        exif_key = {v: k for k, v in ExifTags.TAGS.items()}['Orientation']
        exif = img.getexif()

        if exif_key in exif.keys():
            img = pre.by_pass_exif_orientation(img, exif[exif_key])
        resultOne = predictOne(img)
        return Response(response=resultOne, status=200)
    if (len(data) > 1):
        # base64_form = data[0]
        # base64_data = base64_form.split('base64,')[1]
        # byte_img = io.BytesIO(base64.b64decode(base64_data))

        # data = data.to_dict()
        # data = data.values()
        # imgs = list(data)
        listResultMul = list()
        listItem = list()
        for img in data:
            base64_form = img
            base64_data = base64_form.split('base64,')[1]
            byte_img = io.BytesIO(base64.b64decode(base64_data))
            # img = request.files["image"].read()
            img = Image.open(byte_img)
            rsult = predictOne(img)
            ok = json.loads(rsult)
            for item in ok['files']:
                listResultMul.append(item)

        listResultMul = json.dumps(
            {'files': listResultMul}, sort_keys=True, indent=4, separators=(',', ': ')
        )
        return Response(response=listResultMul, status=200)
    return Response(response=json.dumps({'error': 'ok ok'}), status=400)


@app.route('/', methods=['GET'])
def test():
    print('testing connection: ok')
    return Response(response='testing connection: ok')


@app.route('/health', methods=['GET'])
def health():
    return Response(response='api ok')


@app.route('/health-post', methods=['POST'])
def healthpost():
    return Response(response='post tại sao không dc')


@app.route('/api/mul', methods=['POST'])
def mul():
    img = len(request.files)
    if (img > 1):
        a = request.files.to_dict()
        print(list(a.values())[0])
        return Response(response=json.dumps({'mess': 'xu ly nhieuf'}))
    return Response(response=json.dumps({'mess': 'xu ly mot'}))


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

# if __name__ == '__main__':
#     # app.run(debug=False, host='localhost',
#     #         ssl_context=('./cert.pem', './key.pem'))
#     app.run(debug=False, host='localhost')
#     # wsgi.server(
#     #     eventlet.wrap_ssl(eventlet.listen(('0.0.0.0', 5000)),
#     #                       certfile='cert.pem',
#     #                       keyfile='key.pem',
#     #                       server_side=True), app)


# # serve(app, host='0.0.0.0', port=5000, threads=1,
# #       ssl_context=('cert.pem', 'key.pem'))  # WAITRESS!
