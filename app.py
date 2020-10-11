from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from multiprocessing import Manager, Process
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import RPi.GPIO as GPIO
import numpy as np
import subprocess
import imutils
import pigpio
import signal
import dlib
import time
import cv2
import sys


class Servo:
    def __init__(self, pin):
        self.pin = pin

        self.offset = 0

        subprocess.getstatusoutput('sudo pigpiod')

        self.pi = pigpio.pi()

    def disable(self):
        self.pi.set_servo_pulsewidth(self.pin, 0)

    def setAngle(self, angle):
        new_angle = angle + 90
        self.pi.set_servo_pulsewidth(self.pin, 500 + int(new_angle * (100 / 9)))
        time.sleep(0.2)


class ObjCenter:
    def __init__(self, landmarkPath):
        # load OpenCV's Haar cascade face detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarkPath)

    def update(self, frame, frameCenter):
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect all faces in the input frame
        faces = self.detector(gray, 0)

        targetX, targetY = frameCenter

        # check to see if a face was found
        if len(faces) > 0:

            landmarks = self.predictor(gray, faces[0])

            # unpack the 68 landmark coordinates from the dlib object into a list
            landmarks_list = []
            for i in range(0, landmarks.num_parts):
                landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (255, 255, 255), -1)

            dist = np.sqrt((landmarks.part(21).x - landmarks.part(22).x) ** 2 + (
                    landmarks.part(21).y - landmarks.part(22).y) ** 2)

            face_ptx, face_pty = (int((landmarks.part(21).x + landmarks.part(22).x) / 2),
                                  int((landmarks.part(21).y + landmarks.part(22).y) / 2) - int(dist))

            cv2.circle(frame, (face_ptx, face_pty), 3, (0, 255, 0), -1)
            cv2.circle(frame, (targetX, targetY), 20, (0, 255, 0), 1)

            return (face_ptx, face_pty), faces[0], frame

        return frameCenter, False, frame


class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD

    def initialize(self):
        # intialize the current and previous time
        self.currTime = time.time()
        self.prevTime = self.currTime

        # initialize the previous error
        self.prevError = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0

    def update(self, error, sleep=0.2):
        # pause for a bit
        time.sleep(sleep)

        # grab the current time and calculate delta time
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime

        # delta error
        deltaError = error - self.prevError

        # proportional term
        self.cP = error

        # integral term
        self.cI += error * deltaTime

        # derivative term and prevent divide by zero
        self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0

        # save previous time and error for the next update
        self.prevTime = self.currTime
        self.prevError = error

        # sum the terms and return
        return sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD])


class TouchFree:
    def __init__(self):
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

        # GPIO Mode (BOARD / BCM)
        GPIO.setmode(GPIO.BOARD)

        # set GPIO Pins for Ultrasonic Sensor
        self.GPIO_TRIGGER = 7
        self.GPIO_ECHO = 11

        # set GPIO direction (IN / OUT) for Ultrasonic Sensor
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)

        GPIO.output(self.GPIO_TRIGGER, GPIO.LOW)

        # set GPIO Pins for Relay
        self.GPIO_SPRAY = 16

        # set GPIO direction (IN / OUT) for Relay
        GPIO.setup(self.GPIO_SPRAY, GPIO.OUT)

        GPIO.output(self.GPIO_SPRAY, GPIO.LOW)

        self.spray_duration = 2

        self.distance_threshold = 15

        self.sanitise_img = cv2.imread('images/hand_sanitise.jpg')

        self.mask_threshold = 5

        # define the range for the motors
        self.servoRange = (-90, 90)

        self.panServo = Servo(19)
        self.tiltServo = Servo(12)

        self.cascade = 'shape_predictor_68_face_landmarks.dat'

        self.temp_range = [95, 100]

        self.id = 0

    def start_stream(self):
        self.vs = VideoStream(src=0).start()

        time.sleep(2)

    def get_frame(self):  # fix UI Sizing
        frame = self.vs.read()
        frame = imutils.rotate_bound(frame, 90)

        return frame

    def show_frame(self, final_frame):
        cv2.putText(final_frame, f'#{self.id+1}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('TouchFree', final_frame)
        key = cv2.waitKey(1) & 0xFF

        return key

    def kill_stream(self):

        self.vs.stop()

        del self.vs

    def finish(self, sig, frame):
        self.panServo.disable()
        self.tiltServo.disable()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        self.kill_stream()
        sys.exit()

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

    def ultrasonic_distance(self):

        GPIO.output(self.GPIO_TRIGGER, GPIO.HIGH)

        time.sleep(0.00001)

        GPIO.output(self.GPIO_TRIGGER, GPIO.LOW)

        pulse_end_time = 0
        pulse_start_time = 0

        while GPIO.input(self.GPIO_ECHO) == 0:
            pulse_start_time = time.time()
        while GPIO.input(self.GPIO_ECHO) == 1:
            pulse_end_time = time.time()

        pulse_duration = (pulse_end_time - pulse_start_time)
        distance = round(pulse_duration * 17150, 2)

        return distance

    def spray(self):
        GPIO.output(self.GPIO_SPRAY, GPIO.HIGH)
        time.sleep(self.spray_duration)
        GPIO.output(self.GPIO_SPRAY, GPIO.LOW)

    def spray_sanitizer(self):
        distance = self.ultrasonic_distance()

        if distance <= self.distance_threshold:
            self.spray()
            return True
        else:
            return False

    def get_temperature(self):
        """Find the Temperature using your Sensor"""

        return 99  # Temporary

    def pid_process(self, state, output, p, i, d, objCoord, centerCoord):
        # signal trap to handle keyboard interrupt
        signal.signal(signal.SIGINT, self.finish)

        # create a PID and initialize it
        p = PID(p.value, i.value, d.value)
        p.initialize()

        # loop indefinitely
        while True:
            # calculate the error
            error = centerCoord.value - objCoord.value

            # update the value
            output.value = p.update(error)

            if state.value:
                break

    def in_range(self, val, start, end):
        # determine the input vale is in the supplied range
        return start <= val <= end

    def set_servos(self, state, pan, tlt):
        # signal trap to handle keyboard interrupt
        signal.signal(signal.SIGINT, self.finish)

        # loop indefinitely
        while True:
            # the pan and tilt angles are reversed
            panAngle = -1 * pan.value
            tltAngle = -1 * tlt.value

            # if the pan angle is within the range, pan
            if self.in_range(panAngle, self.servoRange[0], self.servoRange[1]):
                self.panServo.setAngle(panAngle)

            # if the tilt angle is within the range, tilt
            if self.in_range(tltAngle, self.servoRange[0], self.servoRange[1]):
                self.tiltServo.setAngle(tltAngle)

            if state.value:
                break

    def obj_center(self, state, temp, objX, objY, centerX, centerY):
        # signal trap to handle keyboard interrupt
        signal.signal(signal.SIGINT, self.finish)

        self.start_stream()

        # initialize the object center finder
        obj = ObjCenter(self.cascade)

        count = 0

        start_time = 0

        # loop indefinitely
        while True:
            # grab the frame from the threaded video stream
            frame = self.get_frame()

            # calculate the center of the frame as this is where we will
            # try to keep the object
            (H, W) = frame.shape[:2]
            centerX.value = W // 2
            centerY.value = H // 2

            objectLoc = obj.update(frame, (centerX.value, centerY.value))

            ((objX.value, objY.value), rect, final_frame) = objectLoc

            diffX, diffY = centerX.value-objX.value, centerY.value-objY.value

            # display the frame to the screen
            key = self.show_frame(final_frame)

            if rect:

                if -20 < diffX < 20 and -20 < diffY < 20:
                    state.value = 1
                    temp.value = self.get_temperature()
                    if count == 0:
                        start_time = time.time()
                        cv2.imwrite(f'images/{self.id}.jpg', final_frame)

                    while (time.time() - start_time) < 2:

                        cv2.putText(frame, f'Body Temp: {temp.value} F', (100, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        if self.temp_range[0] < temp.value < self.temp_range[1]:
                            cv2.putText(frame, f'Normal Temperature Detected', (0, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f'You Can Proceed', (100, 500),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f'High Temperature Detected', (50, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame, f'You Cannot Proceed', (100, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.show_frame(final_frame)

                    self.kill_stream()

                    break
            else:
                cv2.putText(frame, f'Face Not Detected', (100, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Please Remove Mask', (100, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    def run_subprocesses(self):
        with Manager() as manager:
            # set integer values for the object center (x, y)-coordinates
            centerX = manager.Value("i", 0)
            centerY = manager.Value("i", 0)

            # set integer values for the object's (x, y)-coordinates
            objX = manager.Value("i", 0)
            objY = manager.Value("i", 0)

            # pan and tilt values will be managed by independent PIDs
            pan = manager.Value("i", 0)
            tlt = manager.Value("i", 0)

            # set PID values for panning
            panP = manager.Value("f", 0.09)
            panI = manager.Value("f", 0.08)
            panD = manager.Value("f", 0.002)

            # set PID values for tilting
            tiltP = manager.Value("f", 0.11)
            tiltI = manager.Value("f", 0.10)
            tiltD = manager.Value("f", 0.002)

            state = manager.Value('i', 0)

            temp = manager.Value('f', 0)

            processObjectCenter = Process(target=self.obj_center,
                                          args=(state, temp, objX, objY, centerX, centerY))
            processPanning = Process(target=self.pid_process,
                                     args=(state, pan, panP, panI, panD, objX, centerX))
            processTilting = Process(target=self.pid_process,
                                     args=(state, tlt, tiltP, tiltI, tiltD, objY, centerY))
            processSetServos = Process(target=self.set_servos, args=(state, pan, tlt))

            # start all 4 processes
            processObjectCenter.start()
            processPanning.start()
            processTilting.start()
            processSetServos.start()

            # join all 4 processes
            processObjectCenter.join()
            processPanning.join()
            processTilting.join()
            processSetServos.join()

            processPanning.close()
            processTilting.close()
            processSetServos.close()
            processObjectCenter.close()

            # disable the servos
            self.panServo.disable()
            self.tiltServo.disable()

    def main(self):

        self.start_stream()

        mask_detected = 0
        hand_sanitized = False
        head_align = False
        count = 0
        start_time = 0
        master_frame = None

        self.panServo.setAngle(0)
        self.tiltServo.setAngle(0)

        while True:

            frame = self.get_frame()

            if hand_sanitized:

                if mask_detected >= self.mask_threshold:

                    if count == 0:
                        start_time = time.time()

                    if (time.time() - start_time) > 2:
                        self.kill_stream()
                        self.run_subprocesses()
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

                final_frame = frame

            else:
                final_frame = cv2.resize(self.sanitise_img, (frame.shape[1], frame.shape[0]))

                hand_sanitized = self.spray_sanitizer()

            key = self.show_frame(final_frame)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    def run(self):

        while True:

            self.main()
            self.id += 1


touchfree = TouchFree()
touchfree.run()
