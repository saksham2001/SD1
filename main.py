from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from datetime import datetime
import numpy as np
import time
import cv2


class TouchFree:
    def __init__(self):

        self.send_email = False

        self.sender_email = 'something@something.com'

        self.password = 'your_password'

        self.receiver_email = 'something@something.com'

        self.message = MIMEMultipart("alternative")

        self.message["Subject"] = "Alert: A New Person Entered the Premises"

        self.message["From"] = self.sender_email

        self.message["To"] = self.receiver_email

        # load our serialized face detector model from disk
        self.prototxtPath = r"face_detector/deploy.prototxt"
        
        self.weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)

        # load the face mask detector with face model from disk
        self.maskNet1 = load_model("mask_detector.model")

        # load the face mask detector without face model from disk
        self.maskNet2 = None  # load_model("mask_detector.model")

        self.mask_detected = 0

        self.vs = None

        self.mask_threshold = 2

        self.id = 0

    def start_stream(self):
        self.vs = VideoStream(src=0).start()

        time.sleep(2)

    def get_frame(self):  # fix UI Sizing
        frame = self.vs.read()

        return frame

    def show_frame(self, final_frame):
        cv2.putText(final_frame, f'#{self.id+1}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow(f'TouchFree', final_frame)

        key = cv2.waitKey(1) & 0xFF

        return key

    def finish(self, sig, frame):
        self.vs.stop()
        cv2.destroyAllWindows()

    def detect_face_mask(self, frame):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet1.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return locs, preds

    def detect_mask(self, frame):
        copy_img = frame.copy()

        resized = cv2.resize(copy_img, (254, 254))

        resized = img_to_array(resized)

        resized = preprocess_input(resized)

        resized = np.expand_dims(resized, axis=0)

        preds, _ = self.maskNet2.predict([resized])[0]

        return preds

    def email(self, img_path, mask):
        with open(img_path, 'rb') as f:
            # set attachment mime and file name, the image type is png
            mime = MIMEBase('image', 'png', filename='img1.png')
            # add required header data:
            mime.add_header('Content-Disposition', 'attachment', filename='img1.png')
            mime.add_header('X-Attachment-Id', '0')
            mime.add_header('Content-ID', '<0>')
            # read attachment file content into the MIMEBase object
            mime.set_payload(f.read())
            # encode with base64
            encoders.encode_base64(mime)
            # add MIMEBase object to MIMEMultipart object
            self.message.attach(mime)

        body = MIMEText('''
        <html>
            <body>
                <h1>Alert</h1>
                <h2>A new has Person entered the Premises</h2>
                <h2>Mask: {}</h2>
                <h2>Time: {}</h2>
                <p>
                    <img src="cid:0">
                </p>
            </body>
        </html>'''.format(mask, datetime.now()), 'html', 'utf-8')

        self.message.attach(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(
                self.sender_email, self.receiver_email, self.message.as_string()
            )

    def run(self):

        self.vs = None

        self.start_stream()

        mask_detected = 0
        
        count = 0
        
        start_time = 0

        while True:

            frame = self.get_frame()

            if mask_detected >= self.mask_threshold:
                
                if count == 0:
                    start_time = time.time()

                if (time.time() - start_time) > 4:
                    cv2.imwrite('images/{}.jpg'.format(str(self.id)), frame)
                    if self.send_email:
                        self.email('images/{}.jpg'.format(self.id), 'Wearing')
                    self.id += 1
                    mask_detected = 0
                    break
                else:
                    label1 = 'Mask Detected'
                    color = (0, 255, 0)

                count += 1

            else:
                # detect faces in the frame and determine if they are wearing a
                # face mask or not
                (locs, preds) = self.detect_face_mask(frame)

                if len(preds) > 0:
                    (mask, withoutMask) = preds[0]
                    if mask > withoutMask:
                        mask_detected += 1

                label1 = "No Mask Detected"
                color = (0, 0, 255)

            cv2.putText(frame, label1, (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if label1 == 'No Mask Detected':
                label2 = 'STOP'

                cv2.putText(frame, label2, (100, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                label2 = 'You Can Proceed'

                cv2.putText(frame, label2, (100, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            key = self.show_frame(frame)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                self.finish(None, None)
                break


touchfree = TouchFree()
touchfree.run()
