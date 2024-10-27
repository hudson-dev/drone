from utils.ExampleHelper import get_device_usb_speed_by_port_id
from utils.ExamplePostProcess import post_process_yolo_v5
from djitellopy import Tello

import kp
import cv2
import os
import sys

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame  # it is important to import pygame after that
from pygame import QUIT

class KneronDrone:
    global drone_window, drone_speed, drone_climb, drone_angle, usb_port_id, PWD, sys, MODEL_FILE_PATH, SCPU_FW_PATH, NCPU_FW_PATH, LOOP_TIME, predictions

    # Define the system parameters
    drone_window = 'Smart Drone'
    drone_speed = 30
    drone_climb = 30
    drone_angle = 30

    usb_port_id = 0
    predictions = []

    # Model paths
    PWD = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(1, os.path.join(PWD, '..'))

    MODEL_FILE_PATH = os.path.join(PWD, '/Users/hkim/development/Kneron/drone/res/models/KL720/YoloV5s_640_640_3/models_720.nef')

    LOOP_TIME = 1

    def __init__(self, drone_run, use_controller):
        self.drone_run = drone_run
        self.use_controller = use_controller
        pass

    def init_kneron(self):
        global device_group, model_nef_descriptor, labels

        # Checking to see if KL720 is running at high speed
        try:
            if kp.UsbSpeed.KP_USB_SPEED_HIGH != get_device_usb_speed_by_port_id(usb_port_id=usb_port_id):
                print('\033[91m' + '[Warning] Device is not run at high speed.' + '\033[0m')
        except Exception as exception:
            print('Error: check device USB speed fail, port ID = \'{}\', error msg:[{}]'.format(usb_port_id, str(exception)))
            exit(0)

        # Connect the device
        try:
            print('[Connect Device]')
            device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
            print(' - Success')
        except kp.ApiKPException as exception:
            print('Error: connect device fail, port ID = \'{}\', error msg:[{}]'.format(usb_port_id, str(exception)))
            exit(0)

        # Setting timeout of the usb communication with the device
        print('[Set Device Timeout]')
        kp.core.set_timeout(device_group=device_group, milliseconds=5000)
        print(' - Success')

        # Upload model to device
        try:
            print('[Upload Model]')
            print("MODEL_FILE_PATH: " + MODEL_FILE_PATH);
            
            model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group, 
                                                                file_path=MODEL_FILE_PATH)
            print(' - Success')
        except kp.ApiKPException as exception:
            print('Error: upload model failed, error = \'{}\''.format(str(exception)))
            exit(0)
        
        # Read the class labels
        labels_file = 'coco_name_lists'
        labels_path = os.path.join(os.getcwd(), labels_file)
        labels = open(labels_path).read().strip().split("\n")

    # Initialize DJI Tello drone
    def init_drone(self):
        global drone, frame_read

        drone = Tello()
        drone.connect()
        drone.streamon()
        frame_read = drone.get_frame_read()

        if (self.use_controller): 
            self.init_controller()

    def init_controller(self):
        global joystick, buttonMap

        # Define PS3 controller parameters
        buttonMap = {'UP':4, 'RIGHT':5, 'DOWN':6, 'LEFT':7, 'L2':8, 'R2':9, 'L1':10,
        'R1':11, 'TRIANGLE':12, 'CIRCLE':13, 'CROSS':14, 'SQUARE':15}

        # Initialize PS3 controllers
        pygame.init()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    def keyboard_input(self):
        key = cv2.pollKey()
        print("key: " + str(key))
        if(key == 27):
            pass
        elif (key == ord('t')):
            drone.takeoff()
        elif (key == ord('l')):
            drone.land()
        elif (key == ord('w')):
            drone.move_forward(drone_speed)
        elif (key == ord('s')):
            drone.move_back(drone_speed)
        elif (key == ord('a')):
            drone.move_left(drone_speed)
        elif (key == ord('d')):
            drone.move_right(drone_speed)
        elif (key == ord('q')):
            drone.rotate_counter_clockwise(drone_angle)
        elif (key == ord('e')):
            drone.rotate_clockwise(drone_angle)
        elif (key == ord('r')):
            drone.move_up(drone_climb)
        elif (key == ord('f')):
            drone.move_down(drone_climb)
        else:
            pass

    def controller_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == pygame.JOYBUTTONDOWN:
                if joystick.get_button(buttonMap['R1']):
                    drone.takeoff()
            elif joystick.get_button(buttonMap['R2']):
                self.drone_run = False
                drone.land()
                break
            elif joystick.get_button(buttonMap['L1']):
                drone.move_up(drone_speed)
            elif joystick.get_button(buttonMap['L2']):
                drone.move_down(drone_speed)
            elif joystick.get_button(buttonMap['TRIANGLE']):
                drone.move_forward(drone_speed)
            elif joystick.get_button(buttonMap['CROSS']):
                drone.move_back(drone_speed)
            elif joystick.get_button(buttonMap['CIRCLE']):
                drone.move_right(drone_speed)
            elif joystick.get_button(buttonMap['SQUARE']):
                drone.move_left(drone_speed)
            elif joystick.get_button(buttonMap['UP']):
                drone.move_up(drone_climb)
            elif joystick.get_button(buttonMap['DOWN']):
                drone.move_down(drone_climb)
            elif joystick.get_button(buttonMap['LEFT']):
                drone.rotate_counter_clockwise(drone_angle)
            elif joystick.get_button(buttonMap['RIGHT']):
                drone.rotate_clockwise(drone_angle)

    def run_inference(self):
        print("Running Inference...")

        img = frame_read.frame
        img_bgr565 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2BGR565)

        # Prepare generic image inference input descriptor
        generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
            model_id = model_nef_descriptor.models[0].id,
            inference_number=0,
            input_node_image_list=[
                kp.GenericInputNodeImage(
                    image=img_bgr565,
                    resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                    padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                    normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
                )
            ]
        )
        # start inference work
        for i in range(LOOP_TIME):
            try:
                kp.inference.generic_image_inference_send(device_group=device_group, 
                                                          generic_inference_input_descriptor=generic_inference_input_descriptor)
                
                generic_raw_result = kp.inference.generic_image_inference_receive(device_group=device_group)
            except kp.ApiKPException as exception:
                print(' - Error: inference failed, error = {}'.format(exception))
                exit(0)

        # Retrieve inference node output
        # print('[Retrieve Inference Node Output ]')
        inf_node_output_list = []
        for node_idx in range(generic_raw_result.header.num_output_node):
            inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx = node_idx,
                                                                                             generic_raw_result = generic_raw_result,
                                                                                             channels_ordering = kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)
            inf_node_output_list.append(inference_float_node_output)

        #Post-process the last raw output
        yolo_result = post_process_yolo_v5(
            inference_float_node_output_list = inf_node_output_list,
            hardware_preproc_info = generic_raw_result.header.hw_pre_proc_info_list[0],
            thresh_value = 0.3,
            with_sigmoid=False
        )
        
        print('[Result]')
        print(yolo_result)  
        # Output result image
        # print('[Output Result Image]')
        # print(' - Output bounding boxes on \'{}\''.format(output_img_name))
        for yolo_box_result in yolo_result.box_list:
            b = 100 + (25 * yolo_box_result.class_num) % 156
            g = 100 + (80 + 40 * yolo_box_result.class_num) % 156
            r = 100 + (120 + 60 * yolo_box_result.class_num) % 156
            color = (b, g, r)

            cv2.rectangle(
                img=img,
                pt1=(int(yolo_box_result.x1), int(yolo_box_result.y1)),
                pt2=(int(yolo_box_result.x2), int(yolo_box_result.y2)),
                color=color
            )
            
            class_labels = labels[yolo_box_result.class_num]
            
            cv2.putText(img=img,
                text=class_labels,
                org=(int(yolo_box_result.x1),int(yolo_box_result.y1)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255,255,255),
                thickness=1,
                lineType=cv2.LINE_AA)
            
        cv2.imshow(drone_window,img)