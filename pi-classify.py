"""
Author Matthew Aitchison
Date Jan 2018

A simple example of streamed classification.

"""

import numpy as np
import tensorflow as tf
import cv2
import json
import scipy.ndimage
import cptv
import time

MODEL_NAME = "models/model_lq_thermal"

# globals
session = None
lables = []

def get_session():
    """ Returns a new tensorflow session. """
    return tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))


def get_image_subsection(image, bounds, window_size, boundary_value=None):
    """
    Returns a subsection of the original image bounded by bounds.
    Area outside of frame will be filled with boundary_value.  If None the 10th percentile value will be used.
    """

    # Note: this is a terriable routine, and really needs a rewrite, various libraries have things that perform a
    # similar function to this.

    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]

    # for some reason I write this to only work with even window sizes?
    window_half_width, window_half_height = window_size[0] // 2, window_size[1] // 2
    window_size = (window_half_width * 2, window_half_height * 2)
    image_height, image_width, channels = image.shape

    # find how many pixels we need to pad by
    padding = (max(window_size)//2)+1

    midx = int(bounds.mid_x + padding)
    midy = int(bounds.mid_y + padding)

    if boundary_value is None: boundary_value = np.percentile(image, q=10)

    # note, we take the median of all channels, should really be on a per channel basis.
    enlarged_frame = np.ones([image_height + padding*2, image_width + padding*2, channels], dtype=np.float16) * boundary_value
    enlarged_frame[padding:-padding,padding:-padding] = image

    sub_section = enlarged_frame[midy-window_half_width:midy+window_half_width, midx-window_half_height:midx+window_half_height]

    width, height, channels = sub_section.shape
    if int(width) != window_size[0] or int(height) != window_size[1]:
        print("Warning: subsection wrong size. Expected {} but found {}".format(window_size,(width, height)))

    if channels == 1:
        sub_section = sub_section[:,:,0]

    return sub_section


class Rectangle:
    """ Defines a rectangle by the topleft point and width / height. """
    def __init__(self, topleft_x, topleft_y, width, height):
        """ Defines new rectangle. """
        self.x = topleft_x
        self.y = topleft_y
        self.width = width
        self.height = height

    def copy(self):
        return Rectangle(self.x, self.y, self.width, self.height)

    @property
    def mid_x(self):
        return self.x + self.width / 2

    @property
    def mid_y(self):
        return self.y + self.height / 2

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def overlap_area(self, other):
        """ Compute the area overlap between this rectangle and another. """
        x_overlap = max(0, min(self.right, other.right) - max(self.left, other.left))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return x_overlap * y_overlap

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return "({0},{1},{2},{3})".format(self.left, self.top, self.right, self.bottom)

    def __str__(self):
        return "<({0},{1})-{2}x{3}>".format(self.x, self.y, self.width, self.height)


class Region(Rectangle):
    """ Region is a rectangle extended to support mass. """
    def __init__(self, topleft_x, topleft_y, width, height, mass=0, pixel_variance=0, id=0, frame_index=0):
        super().__init__(topleft_x, topleft_y, width, height)
        # number of active pixels in region
        self.mass = mass
        # how much pixels in this region have changed since last frame
        self.pixel_variance = pixel_variance
        # an identifier for this region
        self.id = id
        # frame index from clip
        self.frame_index = frame_index

    def copy(self):
        return Region(
            self.x, self.y, self.width, self.height, self.mass, self.pixel_variance, self.id, self.frame_index
        )


class Model():

    def __init__(self, filename):

        self.session = get_session()

        self.labels = []

        self.prediction = None
        self.accuracy = None

        self.X = None
        self.y = None
        self.state_out = None
        self.state_in = None

        self.load(filename)

    def load(self, filename):
        """
        Loads a pre-saved model
        """

        saver = tf.train.import_meta_graph(filename + '.meta', clear_devices=True)
        saver.restore(self.session, filename)
        # get additional hyper parameters
        stats = json.load(open(filename + ".txt", 'r'))

        self.labels = stats['labels']

        graph = self.session.graph
        self.prediction = graph.get_tensor_by_name("prediction:0")
        self.accuracy = graph.get_tensor_by_name("accuracy:0")

        # attach to IO tensors
        self.X = graph.get_tensor_by_name("X:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.state_out = graph.get_tensor_by_name("state_out:0")
        self.state_in = graph.get_tensor_by_name("state_in:0")


    def get_feed_dict(self, X, y, state):
        """ returns a feed dictionary for TensorFlow placeholders. """
        return {
            self.X: X,
            self.y: y,
            self.state_in: state
        }

    def classifiy_next_frame(self, frame, state=None):
        """
        Classificy next frame of a tracking window.
        :param frame:
        :param state:
        :return: tuple (prediction, state)
        """
        if state is None:
            state_shape = self.state_in.shape
            state = np.zeros([1, state_shape[1], state_shape[2]], dtype=np.float32)

        batch_X = frame[np.newaxis,np.newaxis,:]
        feed_dict = self.get_feed_dict(batch_X, [0], state)
        pred, state = self.session.run([self.prediction, self.state_out], feed_dict=feed_dict)
        pred = pred[0]
        return pred, state

class VideoClassifier():

    WINDOW_SIZE = 48

    def __init__(self, model):

        self.model = model

        # create a reduced quality optical flow.
        self.opt_flow = cv2.createOptFlow_DualTVL1()
        self.opt_flow.setTau(1 / 4)
        self.opt_flow.setScalesNumber(3)
        self.opt_flow.setWarpingsNumber(3)
        self.opt_flow.setScaleStep(0.5)

        self.prev = None
        self.thermal = None
        self.filtered = None
        self.flow = None
        self.mask = None


    def get_region_channels(self, bounds: Region):
        """
        ...
        """

        # window size must be even for get_image_subsection to work.
        window_size = (max(self.WINDOW_SIZE, bounds.width, bounds.height) // 2) * 2

        # note: thermal and filtered are now the same, they are both thermal zeroed so that the median frame
        # temp is 0
        thermal = filtered = get_image_subsection(self.filtered, bounds,
                                        (window_size, window_size), 0)
        flow = get_image_subsection(self.flow, bounds, (window_size, window_size), 0)
        mask = get_image_subsection(self.mask, bounds, (window_size, window_size), 0)

        if window_size != self.WINDOW_SIZE:
            scale = self.WINDOW_SIZE / window_size
            thermal = scipy.ndimage.zoom(np.float32(thermal), (scale, scale), order=1)
            filtered = scipy.ndimage.zoom(np.float32(filtered), (scale, scale), order=1)
            flow = scipy.ndimage.zoom(np.float32(flow), (scale, scale, 1), order=1)
            mask = scipy.ndimage.zoom(np.float32(mask), (scale, scale), order=0)

            flow *= scale

        # make sure only our pixels are included in the mask.
        mask[mask != bounds.id] = 0
        mask[mask > 0] = 1

        # stack together into a numpy array.
        frame = np.float32(np.stack((thermal, filtered, flow[:, :, 0], flow[:, :, 1], mask), axis=0))

        return frame

    def get_regions_of_interest(self, filtered):

        threshold = np.max(filtered) / 2

        filtered = cv2.blur(filtered, (3, 3))
        filtered[filtered < threshold] = 0
        filtered[filtered > 0] = 1
        filtered = np.uint8(filtered)

        labels, mask, stats, centroids = cv2.connectedComponentsWithStats(filtered)

        padding = 8

        regions = []
        for i in range(1, labels):
            region = Region(
                stats[i, 0] - padding, stats[i, 1] - padding, stats[i, 2] + padding * 2,
                stats[i, 3] + padding * 2, stats[i, 4], 0, i
            )
            regions.append(region)

        return regions, mask

    def process_next_frame(self, thermal, state=None):
        """
        Process the next thermal frame
        :param frame: numpy array of size [H, W]
        :param state:
        :return:
        """

        assert self.model, 'Model must be loaded for classification.'

        profile_times = {}

        start_time = time.time()

        height, width = thermal.shape

        thermal = np.float32(thermal)

        # create a mask
        t0 = time.time()
        self.thermal = thermal.copy()
        self.filtered = self.thermal.copy()
        self.filtered -= (np.median(self.thermal) + 40)
        profile_times['mask'] = time.time() - t0

        # find regions of interest
        t0 = time.time()
        regions, self.mask = self.get_regions_of_interest(self.filtered)
        profile_times['regions of interest'] = time.time() - t0

        # generate optical flow
        t0 = time.time()
        current = np.uint8(np.clip(self.filtered, 0, 255))

        if self.prev is not None:
            # for some reason openCV spins up lots of threads for this which really slows things down, so we
            # cap the threads to 2
            cv2.setNumThreads(2)
            self.flow = self.opt_flow.calc(self.prev, current, self.flow)
        else:
            self.flow = np.zeros([height, width, 2], dtype=np.float32)

        self.prev = current
        profile_times['optical flow'] = time.time() - t0

        # run classifier on center of image
        # note we really should be tracking properly....
        if len(regions) == 0:
            bounds = Region(80-24, 60-24, 48, 48)
        else:
            bounds = regions[0]

        t0 = time.time()
        frame = self.get_region_channels(bounds)
        profile_times['extract channels'] = time.time() - t0

        t0 = time.time()
        prediction, state = self.model.classifiy_next_frame(frame, state)
        profile_times['classify'] = time.time() - t0

        profile_times['total'] = time.time() - start_time

        profile_string = " ".join(["{}:{:.1f}ms".format(k, v * 1000) for k,v in profile_times.items()])
        print(profile_string)

        return bounds, prediction, state

def main():

    t0 = time.time()
    model = Model(MODEL_NAME)
    print("Loaded model {:.1f}s".format(time.time() - t0))

    classifier = VideoClassifier(model)

    print(model.labels)

    video = cptv.CPTVReader(open('resources/20171024-103017-akaroa04.cptv','rb'))
    state = None
    for frame, _ in video:
        bounds, pred, state = classifier.process_next_frame(frame, state)
        print(bounds, pred)



main()
