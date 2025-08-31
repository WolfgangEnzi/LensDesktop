"""
This Python Tool is a Desktop Lens. When executed it will record the screen below it 
and either lens it or de-lens it according to an softened SIE model.

The slider parameters are:

Mask Radius : Radius of a mask for the inverse mode (to cut away lens light that is close to the center).
Position Angle : Position angle that allows to change the orientation of the lens mass profile.
Core Radius : Softening scale / core radius of the power law mass profile.
Axis Ratio : Minor-to-Major axis ratio of the mass profile.
Einstein Radius : Einstein Radius of the mass profile.

You can use the following shortcuts:

Ctrl+S : To Save the currently shown screen (without the GUI printed on top of it).
Ctrl+F : To turn the device camera on/off for recording.
Ctrl+R : To Save a sequence of images in which the Einstein radius increases up
         to its current value (this allows to create nice gifs, e.g. using ffmpeg to postprocess the images).
Ctrl+V : To turn some of the GUI elements on/off.

Use Rightclick to add / remove an RBG circle. This can be used to show Parity of images, magnification and sheer, and conjugate points.
Notice that it matters on which side of the dual view you click when creating this RBG circle, since this will decide the where the circle is anchored to.

I only used this Code on my Macbook where it ran in almost real time. I have not yet confirmed if it is supported by other operating systems.
If it does not work, but you found a fix, please let me know so that we can include it in upcoming versions!

I used ChatGPT o4/o3 to deal with the GUI elements of this Code or to help me figure out how to add the Camera recording and minor things.
If you have any suggestions for how to improve anything or if you want new features to be added, just drop me a message!

The Code in its current form shows some unexpected behaviour for multiple screens when you are not in the main screen.
I plan on fixing this in future versions!

Copyright (c) 2025, Wolfgang Enzi
"""


__author__ = "WolfgangEnzi"

# Import packages
import sys
import gc
import numpy as np
import cv2
import mss
from functools import partial
from scipy.interpolate import RegularGridInterpolator as interp
from scipy.ndimage import zoom
# Set window extent
# base_w = 600
# base_h = 600
ps = 20

import time

CANDIDATES = ("PySide6", "PyQt6", "PyQt5", "PySide2")

def _load_qt():
    for name in CANDIDATES:
        try:
            if name == "PySide6":
                from PySide6 import QtWidgets, QtCore, QtGui
            elif name == "PyQt6":
                from PyQt6 import QtWidgets, QtCore, QtGui
            elif name == "PyQt5":
                from PyQt5 import QtWidgets, QtCore, QtGui
            else:  # PySide2
                from PySide2 import QtWidgets, QtCore, QtGui
            return name, QtWidgets, QtCore, QtGui  # success → bail out
        except ModuleNotFoundError:
            continue
    raise ImportError("No Qt binding found. Install PyQt5/PyQt6/PySide6/PySide2.")

QT_PKG, QtWidgets, QtCore, QtGui = _load_qt()

print(f"[info] Using {QT_PKG}")

# from QtWidgets import QtWidgets.QApplication, QtWidgets.QLabel, QtWidgets.QMainWindow, QtWidgets.QSlider, QtWidgets.QSlider, QtWidgets.QFileDialog, QtWidgets.QShortcut
# from QtCore import Qt, QtCore.QTimer
# from QtGui import QtGui.QImage, QtGui.QPixmap, QtGui.QKeySequence

def capture_cam_rect(cap, width, height):
    """
    Function that captures a frame recorded from the camera of the device
    and then returns it to the lens desktop code cropped and resized.

    Parameters
    ----------
    amp : cap is the cv2 VideoCapture instance
        This is used to catch the current frame of your camera
    width : int
        Pixel width of the LensDesktop window
    height : int
        Pixel height of the LensDesktop window

    Returns
    -------
    frame : numpy.array
        The recorded camera frame cropped and resized
    """

    ret, frame = cap.read()
    frame = frame[:, (frame.shape[1] - frame.shape[0]) // 2: - (frame.shape[1] - frame.shape[0]) // 2 - 1]

    if not ret:
        return np.zeros((height, width, 4)).astype(np.uint8)
    else:
        frame = cv2.resize(frame, (2 * width, 2 * height))[:, :: -1]
        return frame

def capture_screen_rect(sct,mon,x, y, width, height, exclude_window=None):
    """
    Function that captures the screen area defined by (x, y, width, height) 
    while excluding the provided window.

    Parameters
    ----------
    x : int
        X coordinate [pixels] of the rectangle to capture
    y : int
        Y coordinate [pixels] of the rectangle to capture
    width : int
        Pixel width of the LensDesktop window
    height : int
        Pixel height of the LensDesktop window

    Returns
    -------
    frame : numpy.array
        The recorded Desktop frame
    """

    w_full = mon["width"]
    h_full = mon["height"]
    
    raw = sct.grab(mon)
    frame = np.array(raw, dtype=np.uint8)

    top,bottom,left,right= 0,0,0,0
    if x<0:
        left = np.abs(x)
    if x+width > w_full:
        right = np.abs(x+width-w_full)
    if y<0:
        bottom = np.abs(y)
    if y+height> h_full:
        top = np.abs(y+height-h_full)
    xi = x + left
    yi = y + bottom
   
    frame =  np.pad(frame, ((2*bottom, 2*top), (2*left, 2*right), (0,0)), mode='constant', constant_values=0)
    cropped_frame = frame[2*yi:2*yi+2*height,2*xi:2*xi+2*width]

    return cropped_frame

def reduce_points(points, threshold):
    points = [np.array(p) for p in points]
    unique = []

    for p in points:
        if not any(np.linalg.norm(p - u) <= threshold for u in unique):
            unique.append(p)

    return np.array([tuple(p) for p in unique])


def heart_shape(x, y):
    """
    Easteregg Heart-shaped multiplier, to show our love for lensing.

    Parameters
    ----------
    x, y : int
        The x and y position values at which to evaluate the height map that generates a heart shape

    Returns
    -------
    height : numpy.array
        The height of the filter that generates the heart shape of the easteregg 
    """
    z = (0.5 * x**2 + (-1.2 * y + 0.35 - np.sqrt(abs(x * 0.75)))**2)
    height = 1 / (1 + np.exp(5 * (z - 0.3)))
    return height


eps = 1e-5


def SIE_defl(xx, yy, t, s, heart, q, b):
    ct, st = np.cos(t), np.sin(t)
    xv, yv = ct * xx - st * yy, st * xx + ct * yy
    r = np.sqrt(q * q * (xv * xv + s * s) + yv * yv)
    fac = 1.0
    if heart:
        fac = heart_shape(xv, yv)
    A = b * q / np.sqrt(1 - q * q)
    deflx = A * np.arctan(np.sqrt(1 - q * q) * xv / (r + s)) * fac
    defly = A * np.arctanh(np.sqrt(1 - q * q) * yv / (r + q * q * s)) * fac
    deflxv = ct * deflx + st * defly
    deflyv = -st * deflx + ct * defly
    return deflxv, deflyv, r


def create_SIE_map(width, height, b=0.75, q=0.79, s=0.0001, t=0.0, heart=False):
    """
    Function that computes the source positions and deflection angles. 
    When precomputed for fixed parameters, it is possible to map an image from the source plane 
    to the image plane according to a softened Singular Isothermal Ellipsoid (SIE) profile in 
    almost real time.
    Fore reference check https://arxiv.org/pdf/astro-ph/0102341.

    Parameters
    ----------
    width, height : int
        The window width and height determine all the x and y coordinates that are mapped to the source plane.

    Returns
    -------
    map_x, map_y : int
        The map of source positions after the displacement of the softened SIE is applied.
    deflxv, deflyv : int
        The deflection angle maps of the softened SIE.
    xx, yy : int
        The original positions of the grid on the image plane.
    """
    Lx = 2 * width/max(width,height)
    Ly = 2 * height/max(width,height)
    x = np.linspace(-1, 1, width) * Lx/2
    y = np.linspace(-1, 1, height) * Ly/2
    xx, yy = np.meshgrid(x, y)
    deflxv, deflyv, r = SIE_defl(xx  , yy , t, s, heart, q , b)
    xv_new = xx - deflxv
    yv_new = yy - deflyv
    map_x = ((xv_new + Lx/2) * (width - 1)).astype(np.float32)
    map_y = ((yv_new + Ly/2) * (height - 1)).astype(np.float32)
    kappa = 0.5 * b / (1e-30+r * r / q / q) 
    # in the future one could also add the easteregg heartshape to this

    return map_x, map_y, deflxv, deflyv, xx.astype(np.float32), yy.astype(np.float32), kappa


def inverse_remap_image(lensed_img, x_idx, y_idx, mask_radius,base_w,base_h):
    """
    Optimized inverse remapping using flat indexing and manual filling of the 2D histogram 
    that becomes the reconstructed source. When precomputed for fixed parameters, it is possible 
    to map an image back to the source plane according to a softened SIE profile in almost real time.
    Fore reference check https://arxiv.org/pdf/astro-ph/0102341.

    I thank Tian Li for coming up with the name of this mode of the tool, i.e. "Humantonian Monte Carlo".

    Parameters
    ----------
    lensed_img : numpy.array
        Input lensed image that is de-lensed to produce the source reconstruction.
    x_idx, y_idx : int
        The histogram indices obtained from the mapping to the source plane.
    mask_radius=-1 : int
        Radius of the mask that can be used to avoid contamination from lens light.

    Returns
    -------
    inv_img : numpy.array

    """

    h, w, c = lensed_img.shape
    assert c == 3
    inv_img = np.zeros((h * w, 3), dtype=np.float32)
    count = np.zeros((h * w,), dtype=np.float32)

    # Flatten image and indices
    xx, yy = np.meshgrid(np.arange(base_w), np.arange(base_h))
    r2 = (yy.flatten() - base_h // 2) ** 2 + (xx.flatten() - base_w // 2)**2 > mask_radius**2
    flat_img = lensed_img.reshape(-1, 3).astype(np.float32)[r2]
    flat_indices = (y_idx * w + x_idx)[r2]

    # Accumulate values
    np.add.at(inv_img, flat_indices, flat_img)
    np.add.at(count, flat_indices, 1)

    # Avoid division by zero
    mask = count == 0

    if flat_img.size == 0:
        mean_color = np.array([0, 0, 0])
    else:
        mean_color = flat_img.mean(axis=0)

    count[mask] = 1
    inv_img[mask] = mean_color

    inv_img = (inv_img / count[:, None]).reshape(h, w, 3).astype(np.uint8)
    return inv_img

def draw_lens_light(b, kappa, SIE_map_rgb,base_w, base_h):
    # A pseudo lens light distribution. I could have picked a Sersic profile, but this seemed simpler and more directly related to the mass distribution.
    if b > 0:
        img2 = np.zeros((base_h, base_w, 3))
        img2[:, :, 0] = 255
        img2[:, :, 1] = 200
        img2[:, :, 2] = 130
        lkappa = np.log10(1 + kappa)
        alpha = np.clip(lkappa, 0, 1)
        for i in range(3):
            SIE_map_rgb[:, :, i] = SIE_map_rgb[:, :, i] * (1 - alpha) + img2[:, :, i] * alpha
    return SIE_map_rgb

# -------------------------

def exclude_from_capture(widget) -> bool:
    plat = sys.platform
    if plat == 'darwin':
        try:
            from ctypes import c_void_p
            import objc
            from Cocoa import NSWindowSharingNone
            nsview = objc.objc_object(c_void_p=int(widget.winId()))
            nswindow = nsview.window()
            if nswindow is None:
                return False
            nswindow.setSharingType_(NSWindowSharingNone)
            return True
        except Exception:
            return False
    if plat.startswith('win'):
        try:
            import ctypes
            from ctypes import wintypes
            hwnd = int(widget.winId())
            user32 = ctypes.windll.user32
            WDA_EXCLUDEFROMCAPTURE = 0x11  # Win11 22H2+
            res = user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE)
            return bool(res)
        except Exception:
            return False
    return False

class LensDesktop(QtWidgets.QMainWindow):
    '''
    This class is a QtWidgets.QMainWindow of the application that does the lensing of the Desktop / Camera input.
    '''

    def __init__(self):
        '''
        Create a new Window in which the application lives.
        Define all relevant parameters used during the run time of the Script.
        Define the sliders and the initial state of the application.
        '''
        super().__init__()

        self.setWindowTitle("Lens Desktop")

        self.gui_hidden = False
        self.heart = False

        # capture setup
        self.sct = mss.mss()
        try:
            self.mon = self.sct.monitors[0]
            self.w_full = self.mon["width"]
            self.h_full = self.mon["height"]
            self.base_w = self.w_full//4
            self.base_h = self.w_full//4
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Error", f"Invalid monitor index 0")
            sys.exit(1)

        self.fontsize =  self.base_h//42

        self.cam = False
        self.vidcap = None

        self.win_width = self.base_w 
        self.base_h = self.base_h

        self.setWindowFlags(QtCore.Qt.Window)

        # self.setFixedSize(self.win_width, self.base_h)
        self.resize(self.base_w, self.base_h)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(0, 0, self.base_w, self.base_h)

        # Checkboxes.
        # Checkbox to toggle critical curves.
        self.critical_checkbox = QtWidgets.QCheckBox("Critical Curve", self)
        self.critical_checkbox.setGeometry(10, 10, 150, 20)
        self.critical_checkbox.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # Checkbox to toggle dual view.
        self.dual_checkbox = QtWidgets.QCheckBox("Dual view", self)
        self.dual_checkbox.setGeometry(10, 40, 150, 20)
        self.dual_checkbox.stateChanged.connect(self.dual_view_toggled)
        self.dual_checkbox.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # New checkbox to toggle inverse lensing.
        self.inverse_checkbox = QtWidgets.QCheckBox("De-lensing", self)
        self.inverse_checkbox.setGeometry(10, 70, 150, 20)
        self.inverse_checkbox.stateChanged.connect(self.update_view)
        self.inverse_checkbox.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # New checkbox to toggle inverse lensing.
        self.lenslight_checkbox = QtWidgets.QCheckBox("Lens Light", self)
        self.lenslight_checkbox.setGeometry(10, 100, 150, 20)
        self.lenslight_checkbox.stateChanged.connect(self.update_view)
        self.lenslight_checkbox.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # Timer for view updates.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_view)
        refresh_rate_hz = 30  # e.g., 30 frames per second
        interval_ms = int(1000 / refresh_rate_hz)
        self.timer.start(interval_ms)
        self.old_pos = None

        # Sliders for parameters.

        # Slider for the Einstein radius
        self.sliderb = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)

        self.sliderb.setRange(40, 99)
        self.sliderb.setValue(65)
        self.sliderb.valueChanged.connect(self.update_b_value)
        self.b_value = (65 - 40) / (99 - 40)
        self.labelb = QtWidgets.QLabel("Einstein Radius", self)
        self.labelb.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # Slider for the position angle of the lens mass distribution
        self.sliderq = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sliderq.setRange(40, 99)
        self.sliderq.setValue(65)
        self.sliderq.valueChanged.connect(self.update_q_value)
        self.q_value = 0.65
        self.labelq = QtWidgets.QLabel("Axis Ratio", self)
        self.labelq.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # Slider for the softening scale / core radius of the lens mass distribution
        self.sliders = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sliders.setRange(40, 99)
        self.sliders.setValue(41)
        self.sliders.valueChanged.connect(self.update_s_value)
        self.s_value =(41 - 40) / (99 - 40.0)
        self.labels = QtWidgets.QLabel("Core Radius", self)
        self.labels.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # Slider for the position angle of the lens mass distribution
        self.slidert = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slidert.setRange(40, 99)
        self.slidert.setValue(95)
        self.slidert.valueChanged.connect(self.update_t_value)
        self.t_value = (95 - 40) / (99 - 40) * np.pi
        self.labelt = QtWidgets.QLabel("Position Angle", self)
        self.labelt.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")

        # Slider for the Radius of the source mask (only applies for inverse mapping)
        self.slider_mask = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider_mask.setRange(40, 99)
        self.slider_mask.setValue(40)
        self.slider_mask.valueChanged.connect(self.update_mask)
        self.mask_radius = (40 - 40) / (99 - 40) * np.sqrt(self.base_h**2 + self.base_w**2) / 2
        self.label_mask = QtWidgets.QLabel("Mask Radius", self)
        self.label_mask.setStyleSheet(f"font-size: {self.fontsize }pt; font-weight: bold; background-color: lightgray;")
        self.slider_mask.setVisible(False)
        self.label_mask.setVisible(False)

        self.set_Geometry_sliders_and_labels()

        # Shortcuts for several features.

        # Create shortcut 1: Save Screenshot
        save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_screenshot)

        # Create shortcut 2: Valentines day Easteregg
        save_shortcut2 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+L"), self)
        save_shortcut2.activated.connect(self.easteregg)

        # Create shortcut 3: Saving a sequence of einstein radii increasing over time
        save_shortcut3 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        save_shortcut3.activated.connect(self.recording)

        # Create shortcut 4: Change to camera recording
        save_shortcut4 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        save_shortcut4.activated.connect(self.camera_recording)

        # Create shortcut 5: Hide/Show GUI
        save_shortcut5 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+V"), self)
        save_shortcut5.activated.connect(self.HideGUI)

        # Once the above is initialized create and draw the map.
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

        self.ellipses_image_plane = []
        self.ellipses_source_plane = [] 
        
        # try OS-level exclusion
        self.excluded = exclude_from_capture(self)

    def HideGUI(self):

        self.gui_hidden = not self.gui_hidden

        if self.inverse_checkbox.isChecked():
            self.slider_mask.setVisible(not self.slider_mask.isVisible())
            self.label_mask.setVisible(not self.label_mask.isVisible())

        self.slidert.setVisible(not self.slidert.isVisible())
        self.labelt.setVisible(not self.labelt.isVisible())

        self.sliderb.setVisible(not self.sliderb.isVisible())
        self.labelb.setVisible(not self.labelb.isVisible())

        self.sliders.setVisible(not self.sliders.isVisible())
        self.labels.setVisible(not self.labels.isVisible())

        self.sliderq.setVisible(not self.sliderq.isVisible())
        self.labelq.setVisible(not self.labelq.isVisible())

        self.lenslight_checkbox.setVisible(not self.lenslight_checkbox.isVisible())
        self.inverse_checkbox.setVisible(not self.inverse_checkbox.isVisible())
        self.dual_checkbox.setVisible(not self.dual_checkbox.isVisible())
        self.critical_checkbox.setVisible(not self.critical_checkbox.isVisible())

    def set_Geometry_sliders_and_labels(self):

        self.sliderb.setGeometry(130, self.base_h - 40, self.base_w * 4 // 6 , 20)
        self.labelb.setGeometry(10, self.base_h - 40, 110, 15)
        self.sliderq.setGeometry(130, self.base_h - 60, self.base_w * 4 // 6, 20)
        self.labelq.setGeometry(10, self.base_h - 60, 110, 15)
        self.sliders.setGeometry(130, self.base_h - 80, self.base_w * 4 // 6, 20)
        self.labels.setGeometry(10, self.base_h - 80, 110, 15)
        self.slidert.setGeometry(130, self.base_h - 100, self.base_w * 4 // 6, 20)
        self.labelt.setGeometry(10, self.base_h - 100, 110, 15)
        self.slider_mask.setGeometry(130, self.base_h - 120, self.base_w * 4 // 6, 20)
        self.label_mask.setGeometry(10, self.base_h - 120, 110, 15)

    def camera_recording(self):
        self.cam = not self.cam
        if self.cam:
            self.vidcap = cv2.VideoCapture(0)
            if not self.vidcap.isOpened():
                self.cam=False
        else:
            self.vidcap = None

    def update_lensed_map(self):

        self.map_x, self.map_y, self.deflxv, self.deflyv, _, _, self.kappa = create_SIE_map(
            self.base_w, self.base_h,
            b=self.b_value / np.sqrt(self.q_value),
            q=self.q_value,
            s=self.s_value * self.b_value,
            t=self.t_value,
            heart=self.heart
        )

        
        Lx = 2 * self.base_w/max(self.base_w,self.base_h)
        Ly = 2 * self.base_h/max(self.base_w,self.base_h)

        # Flatten the mapping arrays.
        flat_map_x = self.map_x.flatten() / Lx
        flat_map_y = self.map_y.flatten() / Ly

        # Define bins corresponding to pixel boundaries.
        x_bins = np.arange(self.base_w + 1) - 0.5
        y_bins = np.arange(self.base_h + 1) - 0.5

        x_idx = np.digitize(flat_map_x, bins=x_bins) - 1
        y_idx = np.digitize(flat_map_y, bins=y_bins) - 1

        # Clip indices to stay in bounds
        self.x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
        self.y_idx = np.clip(y_idx, 0, len(y_bins) - 2)

        # Save contours for critical curves
        dx = 2.0 / (self.base_w - 1) 
        dy = 2.0 / (self.base_h - 1)
        d_deflx_dx, d_deflx_dy = np.gradient(self.map_x, axis=1) / dx, np.gradient(self.map_x, axis=0) / dy
        d_defly_dx, d_defly_dy = np.gradient(self.map_y, axis=1) / dx, np.gradient(self.map_y, axis=0) / dy
        det = d_deflx_dx * d_defly_dy - d_deflx_dy * d_defly_dx
        crit_mask = (np.sign(det) < 0)
        self.contours, _ = cv2.findContours(crit_mask.astype(np.uint8) , cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )

        
        # Save critical curves that go along
        caustic_contours = []
        for cnt in self.contours:

            if len(cnt) < 2: 
                continue
        
            cnt_caustic = cnt.astype(np.float32, copy=False)
            for i in range(len(cnt_caustic)):
                x_idx = cnt_caustic[i, 0, 0]
                y_idx = cnt_caustic[i, 0, 1]
                norm_x_lensed = -Lx/2 + Lx * x_idx / (self.base_w - 1)
                norm_y_lensed = -Ly/2 + Ly * y_idx / (self.base_h - 1)
                ix = np.clip(int(round(y_idx)), 0, self.base_h - 1)
                jx = np.clip(int(round(x_idx)), 0, self.base_w - 1)
                source_norm_x = norm_x_lensed - self.deflxv[ix, jx]
                source_norm_y = norm_y_lensed - self.deflyv[ix, jx]
                cnt_caustic[i, 0, 0] = (source_norm_x + Lx/2) / Lx * (self.base_w - 1)
                cnt_caustic[i, 0, 1] = (source_norm_y + Ly/2) / Ly * (self.base_h - 1)
            caustic_contours.append(cnt_caustic.astype(np.int32, copy=False))

        self.caustic_curves = caustic_contours

    def dual_view_toggled(self):
        self.update_view()

    # Functions that are called when the sliders are updated

    def update_mask(self, valuem):
        self.mask_radius = (valuem - 40) / (99 - 40) * np.sqrt(self.base_h**2 + self.base_w**2) / 2
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

    def update_b_value(self, valueb):
        self.b_value = (valueb - 40) / (99 - 40.0)
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

    def update_q_value(self, valueq):
        self.q_value = valueq / 100.0
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

    def update_s_value(self, values):
        self.s_value = (values - 40) / (99 - 40.0)
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

    def update_t_value(self, valuet):
        self.t_value = (valuet - 40) / (99 - 40.0) * np.pi
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

    def easteregg(self):
        self.heart = not self.heart
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)

    # Functions for critical curves and caustics:

    def get_critical(self, SIE_map_rgb):
        cv2.drawContours(SIE_map_rgb, self.contours, -1, (255, 255, 255), 5)
        cv2.drawContours(SIE_map_rgb, self.contours, -1, (0, 0, 0), 2)
        return self.contours

    def get_caustics(self, img_unlensed, contours):
        
                
        cv2.drawContours(img_unlensed, self.caustic_curves, -1, (255, 255, 255), 5)
        cv2.drawContours(img_unlensed, self.caustic_curves, -1, (0, 0, 0), 2)

    def save_screenshot(self):
        original_pixmap = self.label.pixmap()
        if not original_pixmap:
            print("No pixmap to save.")
            return

        pixmap = original_pixmap.copy()  # Prevents RuntimeError

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "screenshot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )

        if file_path:
            pixmap.save(file_path)

        gc.collect()

    def recording(self):

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "screenshot.png",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )

        b0 = self.b_value
        K = 160
        brange = np.linspace(0, b0, K)

        for i in range(K):
            self.b_value = brange[i]
            self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius)
            self.update_lensed_map()
            self.update_view()
            original_pixmap = self.label.pixmap()
            pixmap = original_pixmap.copy()  # Prevents RuntimeError

            if file_path:
                pixmap.save(file_path[:-4] + f"_{i:04d}.png")

        gc.collect()

    def showEvent(self, event):
        super().showEvent(event)

    def update_view(self):

        gc.collect()

        # If inverse lensing is selected, do that; otherwise, use single or dual view.
        if self.inverse_checkbox.isChecked():
            if self.gui_hidden is False:
                self.slider_mask.setVisible(True)
                self.label_mask.setVisible(True)
            if self.dual_checkbox.isChecked():
                self.update_inverse_dual_view()
            else:
                self.update_inverse_single_view()
        else:
            self.slider_mask.setVisible(False)
            self.label_mask.setVisible(False)
            if self.dual_checkbox.isChecked():
                self.update_dual_view()
            else:
                self.update_single_view()

    # Forward Updates

    def show_ps(self, img_bgr):
        if len(self.ellipses_image_plane) > 0 or len(self.ellipses_source_plane) > 0:

            if len(self.ellipses_image_plane) > 0:
                x0 = self.ellipses_image_plane[0][0]
                y0 = self.ellipses_image_plane[0][1]

                x_source = int(interp((np.arange(self.base_w), np.arange(self.base_h)), self.map_x.T, method='linear', bounds_error=False, fill_value=None)(np.array([[x0, y0]]))[0])
                y_source = int(interp((np.arange(self.base_w), np.arange(self.base_h)), self.map_y.T, method='linear', bounds_error=False, fill_value=None)(np.array([[x0, y0]]))[0])

            if len(self.ellipses_source_plane) > 0:
                x_source = 2 * (self.ellipses_source_plane[0][0])
                y_source = 2 * (self.ellipses_source_plane[0][1])

            for c in range(3):
                cc = np.zeros((3,))
                cc[c] = 255

                cv2.ellipse(img_bgr,
                            center=(x_source, y_source),
                            axes=(ps, ps),
                            angle=0,
                            startAngle=0 + c * 120,
                            endAngle=120 + c * 120,
                            color=cc,
                            thickness=-1,)
    
    def update_single_view(self):


        self.resize(self.base_w, self.base_h)
        self.label.setGeometry(0, 0, self.base_w, self.base_h)
        
        geo = self.geometry()
        if self.cam:
            arr = capture_cam_rect(self.vidcap, self.base_w, self.base_h)
        else:
            arr = capture_screen_rect(self.sct,self.mon,geo.x(), geo.y(), self.base_w, self.base_h, exclude_window=self)

        img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        self.show_ps(img_bgr)
        
        Lx = 2.0*self.base_w/max(self.base_w,self.base_h)
        Ly = 2.0*self.base_h/max(self.base_w,self.base_h)
        # Forwarz (lensing) remapping.
        SIE_map_bgr = cv2.remap(img_bgr, self.map_x*2.0/Lx, self.map_y*2.0/Ly,
                                interpolation=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT)
        SIE_map_rgb = cv2.cvtColor(SIE_map_bgr, cv2.COLOR_BGR2RGB)

        if self.lenslight_checkbox.isChecked():
            SIE_map_rgb = draw_lens_light(self.b_value, self.kappa, SIE_map_rgb, base_w=self.base_w, base_h=self.base_h)

        # Optionally overlay critical curve.
        if self.critical_checkbox.isChecked():
            self.get_critical(SIE_map_rgb)

        result_image = QtGui.QImage(SIE_map_rgb.data, self.base_w, self.base_h,
                              self.base_w * 3, QtGui.QImage.Format_RGB888)
        result_pixmap = QtGui.QPixmap.fromImage(result_image)

        self.label.setPixmap(result_pixmap)

    def update_dual_view(self):

        # self.setFixedSize(2 * self.base_w, self.base_h)

        self.resize(2 * self.base_w, self.base_h)
        self.label.setGeometry(0, 0, 2 * self.base_w, self.base_h)

        geo = self.geometry()
        if self.cam:
            arr = capture_cam_rect(self.vidcap, self.base_w, self.base_h)
        else:
            arr = capture_screen_rect(self.sct,self.mon,geo.x(),geo.y(), self.base_w, self.base_h, exclude_window=self)

        img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        self.show_ps(img_bgr)

        img_unlensed = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_unlensed.shape[0] != self.base_h or img_unlensed.shape[1] != self.base_w:
            img_unlensed = cv2.resize(img_unlensed, (self.base_w, self.base_h))


        Lx = 2*self.base_w/max(self.base_w,self.base_h)
        Ly = 2*self.base_h/max(self.base_w,self.base_h)
        SIE_map_bgr = cv2.remap(img_bgr, self.map_x*2/Lx, self.map_y*2/Ly,
                                interpolation=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT)
        SIE_map_rgb = cv2.cvtColor(SIE_map_bgr, cv2.COLOR_BGR2RGB)

        if self.lenslight_checkbox.isChecked():
            SIE_map_rgb = draw_lens_light(self.b_value, self.kappa, SIE_map_rgb, base_w=self.base_w, base_h=self.base_h)

        if self.critical_checkbox.isChecked():
            contours = self.get_critical(SIE_map_rgb)
            self.get_caustics(img_unlensed, contours)

        # Combine views.
        combined = np.hstack((img_unlensed, SIE_map_rgb))
        result_image = QtGui.QImage(combined.data, 2 * self.base_w, self.base_h,
                              2 * self.base_w * 3, QtGui.QImage.Format_RGB888)
        result_pixmap = QtGui.QPixmap.fromImage(result_image)
        self.label.setPixmap(result_pixmap)

    # Inverse Updates
    def inverse_ps_show(self, img_bgr):
        if len(self.ellipses_image_plane) > 0 or len(self.ellipses_source_plane) > 0:

            if len(self.ellipses_source_plane) > 0:

                if self.base_h == self.base_w: # Right now only supported for the equal dimensions case

                    id = np.array(np.where((self.x_idx.reshape((self.base_w, self.base_h)) - self.ellipses_source_plane[0][0])**2
                                        + (self.y_idx.reshape((self.base_w, self.base_h)) - self.ellipses_source_plane[0][1])**2 < 1.0))

                    id = reduce_points(id.T, ps * 2).T

                    if len(id) > 0:
                        for i in range(len(id[0])):
                            x0, y0 = id[1][i], id[0][i]

                            for c in range(3):
                                cc = np.zeros((3,))
                                cc[c] = 255

                                cv2.ellipse(img_bgr,
                                            center=(x0, y0),
                                            axes=(ps // 2, ps // 2),
                                            angle=0,
                                            startAngle=0 + c * 120,
                                            endAngle=120 + c * 120,
                                            color=cc,
                                            thickness=-1,)

            if len(self.ellipses_image_plane) > 0:
                x0 = self.ellipses_image_plane[0][0]
                y0 = self.ellipses_image_plane[0][1]

                for c in range(3):
                    cc = np.zeros((3,))
                    cc[c] = 255

                    cv2.ellipse(img_bgr,
                                center=(x0, y0),
                                axes=(ps // 2, ps // 2),
                                angle=0,
                                startAngle=0 + c * 120,
                                endAngle=120 + c * 120,
                                color=cc,
                                thickness=-1,)

    def update_inverse_single_view(self):
        """
        When inverse lensing is enabled, we capture the image, perform the forward mapping as before,
        and then use our inverse_remap_image() routine to “undo” the lensing.
        """
        # self.setFixedSize(self.base_w, self.base_h)

        self.resize(self.base_w, self.base_h)
        self.label.setGeometry(0, 0, self.base_w, self.base_h)

        geo = self.geometry()
        if self.cam:
            arr = capture_cam_rect(self.vidcap, self.base_w, self.base_h)
        else:
            arr = capture_screen_rect(self.sct,self.mon,geo.x(), geo.y(), self.base_w, self.base_h, exclude_window=self)

        img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if img_bgr.shape[0] != self.base_h or img_bgr.shape[1] != self.base_w:
            img_bgr = cv2.resize(img_bgr, (self.base_w, self.base_h))

        self.inverse_ps_show(img_bgr)

        # Now, perform the inverse remapping to (attempt to) recover the original.
        inv_img = self.inv_map(img_bgr)

        if self.critical_checkbox.isChecked():
            contours = self.get_critical(img_bgr)
            self.get_caustics(inv_img, contours)

        result_image = QtGui.QImage(inv_img.data, self.base_w, self.base_h,
                              self.base_w * 3, QtGui.QImage.Format_RGB888)
        result_pixmap = QtGui.QPixmap.fromImage(result_image)
        self.label.setPixmap(result_pixmap)

    def update_inverse_dual_view(self):
        """
        When inverse lensing is enabled, we capture the image, perform the forward mapping as before,
        and then use our inverse_remap_image() routine to “undo” the lensing.
        """

        # self.setFixedSize(2 * self.base_w, self.base_h)

        self.resize(2 * self.base_w, self.base_h)
        self.label.setGeometry(0, 0, 2 * self.base_w, self.base_h)

        geo = self.geometry()
        if self.cam:
            arr = capture_cam_rect(self.vidcap, self.base_w, self.base_h)
        else:
            arr = capture_screen_rect(self.sct,self.mon,geo.x(), geo.y(), self.base_w, self.base_h, exclude_window=self)

        img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        if img_bgr.shape[0] != self.base_h or img_bgr.shape[1] != self.base_w:
            img_bgr = cv2.resize(img_bgr, (self.base_w, self.base_h))

        self.inverse_ps_show(img_bgr)

        xx, yy = np.meshgrid(np.arange(self.base_w), np.arange(self.base_h))
        r2 = (yy - self.base_h // 2) ** 2 + (xx - self.base_w // 2)**2 > self.mask_radius**2
        img_bgr[r2 == False] = img_bgr.mean()

        # Now, perform the inverse remapping to (attempt to) recover the original.
        inv_img = self.inv_map(img_bgr)

        if self.critical_checkbox.isChecked():
            contours = self.get_critical(img_bgr)
            self.get_caustics(inv_img, contours)

        # Combine views.
        combined = np.hstack((img_bgr, inv_img))
        result_image = QtGui.QImage(combined.data, 2 * self.base_w, self.base_h,
                              2 * self.base_w * 3, QtGui.QImage.Format_RGB888)
        result_pixmap = QtGui.QPixmap.fromImage(result_image)
        self.label.setPixmap(result_pixmap)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.old_pos = event.globalPos()

        if event.button() == QtCore.Qt.RightButton:
            # I thank Christopher Pattison and Sergi Sirera Lahoz for this nice idea!

            if len(self.ellipses_image_plane) > 0 or len(self.ellipses_source_plane) > 0:
                self.ellipses_image_plane = []
                self.ellipses_source_plane = []
                gc.collect()
            else:

                ex, ey = event.x(), event.y()

                if self.inverse_checkbox.isChecked():
                    if self.dual_checkbox.isChecked():
                        # check for left and right side add accordingly
                        if ex > self.base_w:
                            self.ellipses_source_plane += [( (ex - self.base_w), ey,)]
                        else:
                            self.ellipses_image_plane += [(ex, ey,)]
                    else:
                        self.ellipses_source_plane += [(ex, ey,)]

                else:
                    if self.dual_checkbox.isChecked():
                        if ex > self.base_w:
                            self.ellipses_image_plane += [( (ex - self.base_w), ey,)]
                        else:
                            self.ellipses_source_plane += [(ex, ey,)]
                    else:
                        self.ellipses_image_plane += [(ex, ey,)]
                
                self.update_lensed_map()

    def resizeEvent(self, event):
        # This is called whenever the window is resized
        if self.dual_checkbox.isChecked():
            self.base_w = self.width()//2
            self.base_h = self.base_w#self.height()
        else:
            self.base_w = self.width()
            self.base_h = self.base_w#self.height()
        
        self.update_lensed_map()
        self.inv_map = partial(inverse_remap_image, x_idx=self.x_idx, y_idx=self.y_idx, mask_radius=self.mask_radius, base_w=self.base_w, base_h=self.base_h)
        self.set_Geometry_sliders_and_labels()
        # Always call the parent implementation
        super().resizeEvent(event)


    def mouseMoveEvent(self, event):

        if self.old_pos is not None:
            delta = event.globalPos() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

    def closeEvent(self, event):
        event.accept()
        QtWidgets.QApplication.instance().quit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    lens_desktop = LensDesktop()
    lens_desktop.show()
    sys.exit(app.exec_())
