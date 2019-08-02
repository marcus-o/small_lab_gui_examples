# small_lab_gui_examples
This repository contains exemplary single-user GUIs for small measurements and laboratory control. The goal of these is not to provide a fully integrated framework for creating GUIs, but some examples of how to use bokeh to create small programs acquiring and displaying data. The code is far from optimal and only implements a small subset of bokeh's and most devices' functions. It is provided as is without any warranty. I will answer questions sent to marcus.ossiander@gmail.com.

## tof
A program that performs a pump-probe experiment by driving a piezo and reading out the results of a time-of-flight spectrometer digitized via a time-to-digital converter.

## lockin
A program that performs a pump-probe experiment by driving a piezo and reading out the results of a lock-in amplifier.

## whitelight
A program that collects (non scanning) white-light interferometer data from a spectrometer and calculates the group delay difference of the two interferometer arms.

## greateyes
A program that performs a pump-probe experiment by driving a piezo and reading out a camera picture. Currently in untested state.

## movement
A gui for driving the stepper motors in an experiment incrementally or to pre-defined positions.

# small_lab_gui components

## helper
Provides classes to quickly setup and a run a single user gui and perform a measurement

### helper.bokeh_gui_helper
Provides class bokeh_gui_session_handler, which allows to quickly create and open a single user GUI from a python script by overloading the function bokeh_gui_session_handler.open_session() (for an example see whitelight.py at the very bottom).

Provides class running, which allows to track if a specific or any measurement is running in the current program.

### helper.measurement
Provides class measurement, which allows to run an infinitely of finitely acquire measurements in an external thread (such that it can be blocking without blocking the gui) and frequently updates the main gui in a way that is compatible with bokeh.

### helper.bokeh_plot_helper
Allows simplified plotting without explicitly taking care of data sources, only useful for very simple data, otherwise using bokeh directly offers vastly more possibilities.

### helper.fourierAndAxis
Performs an FFT/DFT of electric fields and returns the correct frequency / time axis, one or two sided spectra and phase / group delay for convenience.

### helper.posToELOG
Allows to automatically post to post to psi elog labbooks (https://elog.psi.ch/elog/). Limited possibilities, but could be easily expanded.

## digitizers
Wrapper classes for hardware acquiring data.

### digitizers.digitizer
Base structure for a digitizer object, every such should implement a constructor and functions setup, frame, stop and close. Additionally, function readout_continuous can be implemented for quick data acquisition without checks for the time or sequence the data was taken at, e. g. for camera readout during alignment. Five different devices (Zurich Instruments lock-in, Oceanoptics spectrometer, Pfeiffer pressure gauge, Fast Comtech time-to-digital converter and Greateyes camera (for the latter two, no official python dll wrappers exist to my knowledge)) are available as examples. For every device, a dummy device producing random data in the same format exists for out-of-the-lab testing.

## axes
Wrapper classes for hardware moving stuff.

### axes.linear_axis
Base structure for a linearly moving axis and controller. First intialize a controller that handles the connection and then add one or more axes. Examples for a piezo from Piezojena, stepper drivers from Physikinstrumente and from Micos.

### axes.toggle
Base structure for a moving device with two states. First intialize a controller that handles the connection and then add one or more toggles. Example for a Conrad delay card.

# todo currently
* Add doc strings
* Add minimal examples
* Add possibility to simultaneously acquire data in measurement
* Change sleeps in devices to asynchronus sleeps / avoid blocking async flow

