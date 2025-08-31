# Important Disclaimer

This tool uses screen / camera recording to map images in (almost) real time to a lensed/delensed image.

In order for the tool to apply to other windows you will need to give it the rights for screen / camera recording. I, therefore, recommend to use pyinstaller to create an ".app" file first. Once you have that you can give these rights to the app rather than the terminal (the latter might being a security concern). Your computer might close the app before you can give permission for those recordings, but you usually just have to restart it when that happens.

I only used this Code on my Macbook where it ran in almost real time. I have not yet confirmed if it is supported by other platforms.
If it does not work for those, but you found a fix, please let me know so that we can include it in upcoming versions!

# Installation

First download the code into a folder of your choice:
```
git clone https://github.com/WolfgangEnzi/LensDesktop.git
cd LensDesktop
```

I also recommend to create a new environment either with conda or venv:

```
conda create -n lens_desktop python=3.11
conda activate lens_desktop
```

or

```
python -m venv lens_desktop
source lens_desktop/Scripts/activate
```

The following commands should create an executable (.exe or .app) in a newly created dist folder. You can grant the executable the rights for screen / camera recording without having to give these rights to the terminal.

You can then install the required packages. 
For Macbooks users this should be:

```
python -m pip install pyobjc mss pyinstaller pyobjc-framework-Quartz numpy scipy opencv pyqt 
```

Alternatively for Windows users:

```
pip install numpy scipy opencv-python pyqt5 mss pyinstaller 
```

Using pyinstaller you can then create the executable in a new folder:

```
pyinstaller --noconfirm LensDesktop.spec
```

The first start after creating the app usually takes a bit longer.

# Lens Desktop

This Python Tool is a Desktop Lens. When executed it will record the screen below it 
and either lens it or de-lens it according to an softened SIE model.

The slider parameters are:

Mask Radius : Radius of a mask for the inverse mode (to cut away lens light that is close to the center).
Position Angle : Position angle that allows to change the orientation of the lens mass profile.
Core Radius : Softening scale / core radius of the power law mass profile.
Axis Ratio : Minor-to-Major axis ratio of the mass profile.
Einstein Radius : Einstein Radius of the mass profile.

You can use the following shortcuts:

- Ctrl+S : To Save the currently shown screen (without the GUI printed on top of it).
- Ctrl+F : To turn the device camera on/off for recording.
- Ctrl+R : To Save a sequence of images in which the Einstein radius increases up
         to its current value (this allows to create nice gifs, e.g. using ffmpeg to postprocess the images).
- Ctrl+V : To turn some of the GUI elements on/off.

Use Rightclick to add / remove an RBG circle. This can be used to show Parity of images, magnification and sheer, and conjugate points.
Notice that it matters on which side of the dual view you click when creating this RBG circle, since this will decide the where the circle is anchored to.

I used ChatGPT o4/o3 to deal with the GUI elements of this Code and to help me figure out how to include the Camera recording and minor things.
If you have any suggestions for how to improve anything or if you want new features to be added, just drop me a message!

The Code in its current form shows some unexpected behaviour for multiple screens when you are not in the main screen.
I plan on fixing this in future versions!

Copyright (c) 2025, Wolfgang Enzi
