import numpy as np
import picamera
import time
import os
import io
import cv2
from compute_features import compute_features as _cf
from compute_features import detect as _detect


class BeeCamera(picamera.PiCamera):
    def __init__(self, resolution=(1280, 960), framerate=10,
                 pre_event_time=60, post_event_time=60,
                 storage='/home/pi/Videos/', draw_rect=True,
                 bg_decay=0.1, bg_framerate=1, bg_images=1):
        """Wrapper for the picamera that detects bumblebees.

        Parameters
        ----------
        resolution : Array like
            Array like object with length two. Sets the resolution
            of the camera
        framerate : int
            The framerate of the video.
        pre_event_time : int
            No. of seconds of video to save before event happened.
        post_event_time : int
            No. of seconds of video to save after event happened.
        storage : str
            The directory to save images and videos.
        draw_rect : bool
            Wether or not to draw rectangles around all detected
            bumblebees.
        bg_decay : float
            The decay constant for the background computations.
        bg_framerate : int or float
            The framerate that is used to form initial background.
        bg_images : int
            Total no. of images used to form initial background.
        """
        super(BeeCamera, self).__init__(
            resolution = resolution,
            framerate = framerate
        )
        self.zoom = (0.15, 0.3, .8, .65)
        
        # Set instance variables.
        self.video_buffer = picamera.PiCameraCircularIO(
            camera = self,
            seconds = pre_event_time + post_event_time
        ) 
        self.background = self.create_background(bg_framerate, bg_images)
        self.storage = storage
        if not os.path.exists(storage):
            os.mkdir(storage)
        self.pre_event_time = pre_event_time
        self.post_event_time = post_event_time
        self.image = self.capture_image()
        self.draw_rect = draw_rect

        # Start video recording.
        self.start_recording(self.video_buffer, format='h264')
        self.wait_recording(2)
	self.decay = bg_decay

        # Create object detector
	self.bumblebee_detector = BumblebeeDetector(
            filter_area=True,
            min_area=10,
            filter_circ=False,
            filter_convex=False
        )


    def set_storage(self, storage):
        self.storage = storage
        if not os.path.exists(storage):
            os.mkdir(storage)

    def reset_video_buffer(self):
        """Create a new video buffer to record at.
        """
        self.stop_recording()
        self.video_buffer = picamera.PiCameraCircularIO(
            camera = self,
            seconds = self.pre_event_time + self.post_event_time
        )
        self.start_recording(self.video_buffer, format='h264')
        self.wait_recording(2)
    

    def set_pre_event_time(self, pre_event_time):
        """Set how long video to save before event.
        """
        self.pre_event_time = pre_event_time
        self.reset_video_buffer()


    def set_post_event_time(self, post_event_time):
        """Set how long video to save after event.
        """
        self.post_event_time = post_event_time
        self.reset_video_buffer()


    def set_event_time(self, pre_event_time, post_event_time):
        """Set how long video to save before and after event.
        """
        self.pre_event_time = pre_event_time
        self.post_event_time = post_event_time
        self.reset_video_buffer()
    

    def capture_image(self, fast_capture=True):
        """Capture an image.

        Parameters
        ----------
        fast_capture : bool
            Whether or not the video port is used to capture the image.
            The image quality is reduced but capture time is massively
            increased when this is true.

        Returns
        -------
        image : numpy.ndarray
            The image captured from the camera, last dimension is channels.
        """
        stream = io.BytesIO()
        self.capture(stream, format='rgb', use_video_port=fast_capture)
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        return data.reshape(self.resolution[1], self.resolution[0], 3)


    def update_image(self):
        """Updates this camera's image.
        """
        self.image = self.capture_image()
        self.update_background()


    def save_video(self, name='mov.h264', folder='videos/', printout=True):
        """Save current video to disk.

        The current video stream is saved to disk. Standard path
        is this camera's storage variable. Subfolder can be specified
        by the folder parameter and the name of the file is specified
        by the name parameter.
        
        Parameters
        ----------
        name : str
            Name of the video file
        folder : str
            Subfolder to place video in.
        """
        
        path = self.storage + folder
        if not os.path.exists(path):
            os.mkdir(path)
        
        with io.open(path + name, 'wb') as output:
            for frame in self.video_buffer.frames:
                if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                    self.video_buffer.seek(frame.position)
                    break
            while True:
                buf = self.video_buffer.read1()
                if not buf:
                    break
                output.write(buf)
        # Wipe the circular stream once we're done
        if printout:
            print('Saved video to: ' + path + name)
        self.video_buffer.seek(0)
        self.video_buffer.truncate()


    def save_image(self, name='img.jpg', folder='images/',
                   printout=True, image=None):
        """Save this camera's current image to disk as a jpeg file.

        Current image is is saved to disk. Standard path is this
        camera's storage variable. Subfolder can be specified by
        the folder parameter and the name of the file is specified
        by the name parameter.

        Parameters
        ----------
        name : str
            Name of the image file.
        folder : str
            Subfolder to place image in.
        printout : bool
            Wether to print out information about the save
        image : np.ndarray
            Optional, if some other image should be saved pass that here.
        """       
        path = self.storage + folder
        if not os.path.exists(path):
            os.mkdir(path)
        image = self.image if image is None else image
        cv2.imwrite(path + name, image.astype(np.uint8))
        if printout:
            print('Saved image to: ' + path + name)


    def create_background(self, framerate=1, num_images=10, debug=False):
        """Create initial background.

        Parameters
        ----------
        framerate : int or float
            Framerate for the image cascade used to create initial background
        num_images : int
            Total no. of images used to create initial background.
        debug : bool
            If this is true, the array of images, not its mean is returned.
        
        Returns
        -------
        background : numpy.ndarray
            The background at current time.
        """
        image_array = np.zeros([num_images, self.resolution[1], self.resolution[0], 3])
        curr_time = time.time()
        for i in range(num_images):
            # Make sure enough time has passed since last image.
            prev_time = curr_time
            curr_time = time.time()
            if curr_time - prev_time < 1./framerate:
                time.sleep(1./framerate - (curr_time - prev_time))

            # Capture Image.
            image_array[i, :, :, :] = self.capture_image()

        # Create background.
        background = np.mean(image_array, axis=0)
        return background if not debug else image_array


    def new_background(self, framerate=1, num_images=10):
        """Sets a new background for the camera.

        Parameters
        ----------
        framerate : int or float
            Framerate for the image cascade used to create initial background
        num_images : int
            Total no. of images used to create initial background.
        """
        self.background = self.create_background(framerate, num_images)
    

    def update_background(self):
        """Update image using an exponential decaying running average.

        Update the background using an exponential decaying running average,
        that is:
        .. math:: \bar{x}_{n+1} = \delta x_{n+1} + (1-\delta) \bar{x}_n,

        where :\math:`x_{i}` is the i-th image, :math:`\bar{x}_i` is
        the average after i iterations and :math:`\delta` is the
        decay constant deciding how much influence the current image
        has on the mean.
        """
        self.background = self.decay*self.image + (1-self.decay)*self.background
      

    def detect_bumblebee(self, video_path='videos/', image_path='images/'):
        """Check wether or not there is a bumblebee in front of the bee hive.

        Saves current frame and video if there is a bumblebee present.

        Parameters
        ----------
        video_path : str
            The subpath that video files are stored in.
        image_path : str
            The subpath that images are stored in.
        """
        self.update_image()
        
        bumblebee_present = self.bumblebee_detector.detect(
            frame=self.image,
            draw_rectangles=self.draw_rect
        )
        
        if bumblebee_present:
            timestamp = time.localtime()
            name_prefix = str(timestamp.tm_year) + '-' + str(timestamp.tm_mon)+ '-' \
                        + str(timestamp.tm_mday) + '-h' + str(timestamp.tm_hour) + '-m' \
                        + str(timestamp.tm_min)
            self.wait_recording(self.post_event_time)
            self.save_video(name=name_prefix+'_video.h264')
            self.save_image(name=name_prefix+'_image.jpg')
            self.new_background()


    def record_test_footage(self, sense=None):
        if not os.path.exists('test_no.dat'):
            test_no = 1
            with open('test_no.dat', 'w') as f:
                f.write(str(test_no))
        else:
            with open('test_no.dat', 'r') as f:
                test_no = int(f.readline())+1
            with open('test_no.dat', 'w') as f:
                f.write(str(test_no))
        if sense is not None:
            W = [255, 255, 255]
            sense.show_message(str(test_no))
            sense.set_pixels([W]*64)

        curr_time = time.time()
        for i in range(self.pre_event_time+self.post_event_time):
            self.update_image()
            
            self.save_image(
                name='{:03d}_image.jpg'.format(i),
                folder='test_{:03d}_images/'.format(test_no),
                printout=False
            )
            self.save_image(
                name='{:03d}_background.jpg'.format(i),
                folder='test_{:03d}_backgrounds/'.format(test_no),
                printout=False,
                image=self.background
            )
            
            prev_time = curr_time
            curr_time = time.time()
            if curr_time - prev_time < 1:
                time.sleep(1 - (curr_time - prev_time))
        self.save_video(name='test.h264', folder='test_{:03d}_videos/'.format(test_no), printout=False)
      


class BumblebeeDetector(object):
    def __init__(self, 
                 filter_area=None, min_area=None, max_area=None,
                 filter_circ=None, min_circ=None, max_circ=None,
                 filter_convex=None, min_convex=None, max_convex=None):
        """Constructor for the a bumblebee detector object.

        The parameters are used for the OpenCV blob detector. Standard
        values are used if set to None.
        """
        blob_params = cv2.SimpleBlobDetector_Params()
        
        blob_params.filterByArea = filter_area if filter_area is not None else blob_params.filterByArea
        blob_params.minArea = min_area if min_area is not None else blob_params.minArea
        blob_params.maxArea = max_area if max_area is not None else blob_params.maxArea

        blob_params.filterByCircularity = filter_circ if filter_circ is not None else blob_params.filterByCircularity
        blob_params.minCircularity = min_circ if min_circ is not None else blob_params.minCircularity
        blob_params.maxCircularity = max_circ if max_circ is not None else blob_params.maxCircularity

        blob_params.filterByConvexity = filter_convex if filter_area is not None else blob_params.filterByConvexity
        blob_params.minConvexity = min_convex if min_convex is not None else blob_params.minConvexity
        blob_params.maxConvexity = max_convex if max_convex is not None else blob_params.maxConvexity

        blob_params.filterByInertia = False
        blob_params.filterByColor = False

	self.detector = cv2.SimpleBlobDetector(blob_params)


    def detect(self, frame, thresh=50, draw_rectangles=True):
        """Check whether or not a bumblebee is in given frame.

        Parameters
        ----------
        frame : np.ndarray
          Current video frame.
        bg : np.ndarray
          Background image to be subtracted from the frame.
        draw_rectangles : bool
          Wether or not to draw rectangles around the detected bumblebees.
        """
        th_err = cv2.threshold(
            src = frame.mean(axis=2).astype(np.uint8),
            thresh = thresh,
            maxval = 255,
            type = cv2.THRESH_BINARY_INV
        )[1]

        # Find blobs:
        keypoints = self.detector.detect(th_err)
        th_err2 = np.array(th_err)

        # Find ROI's
        mask = np.zeros(np.array(th_err.shape)+2, dtype=np.uint8)
        blobs = [0]*len(keypoints)
        areas = [0]*len(keypoints)
        for i, k in enumerate(keypoints):
            x = int(k.pt[0])
            y = int(k.pt[1])
            areas[i], blobs[i] = cv2.floodFill(th_err2, mask*0, (x, y), i+1)

        blobs = np.array(blobs)

        # Check if any of the ROIs contain a bumblebee
        return_val = False
        rect_list = []
        for i in range(len(keypoints)):
            x_min = blobs[i, 0]
            x_max = x_min + blobs[i, 2]
            y_min = blobs[i, 1]
            y_max = y_min + blobs[i, 3]
            roi = frame[y_min:y_max, x_min:x_max]
            if self.detect_single_bumblebee(roi):
                print('Detected bumblebee')
                if draw_rectangles:
                    rect_list.append(((x_min-3, y_min-3), (x_max+2, y_max+2)))
                return_val = True

        # Draw rectangles
        for rect in rect_list:
            cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), 1)
        return return_val


    def detect_single_bumblebee(self, blob):
        """Check wether or not the given blob contains a single bumblebee.
        """
        return self.class_score(blob) >= 0


    def compute_features(self, blob):
        """Compute the image features of given blob.
        """
        return _cf(blob)[0]
    

    def class_score(self, blob):
        """Classify a blob.
        """
        return _detect(self.compute_features(blob))


