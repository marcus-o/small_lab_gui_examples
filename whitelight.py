import numpy as np
import time
from functools import partial
import os
import datetime
import copy
import bokeh
import h5py

from small_lab_gui.helper import bokeh_gui_helper as bgh
from small_lab_gui.helper import measurement
from small_lab_gui.helper import fourierAndAxis

testing = True
if not testing:
    # for the lab
    from small_lab_gui.digitizers.digitizer_oceanoptics_spectrometer \
        import oceanoptics_spectrometer
else:
    # for testing
    from small_lab_gui.digitizers.digitizer_oceanoptics_spectrometer_dummy \
        import oceanoptics_spectrometer_dummy as oceanoptics_spectrometer


class whitelight_alignment_gui():
    def __init__(self, doc, running, spectrometer):
        self.title = 'Alignment'
        self.lastrefresh = 0

        # measurement thread
        self.cont_readout = None
        # bokeh doc for callback
        self.doc = doc
        # digitizer card
        self.spectrometer = spectrometer
        # global measurement running indicator
        self.running = running
        # autoscale variable

        # Set up widgets
        self.startBtn = bokeh.models.widgets.Button(
            label='Start', button_type='success', disabled=False)
        self.integrationInput = bokeh.models.widgets.TextInput(
            title='Integration time [msec]', value='13')

        self.linePlot = bgh.plot_2d()
        self.linePlot.line(legend='Current', line_color='red')

        # save button
        self.saveBtn = bokeh.models.widgets.Button(
            label='Save Spectrum', button_type='success', disabled=False)
        self.saveBtn.on_click(self.save_spectrum)
        self.saveNameInput = bokeh.models.widgets.TextInput(
            title='Legend Name', value='Name')

        # arrange layout
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.integrationInput,
            self.saveBtn, self.saveNameInput)
        self.layout = bokeh.layouts.row(
            self.inputs, self.linePlot.element, width=800)

        # start thread callback
        self.startBtn.on_click(self.start)

    def start(self):
        # in case this measurement is running, stop it
        if self.running.am_i_running(self):
            self.stop()
        # in case a different measurement is running, do nothing
        elif self.running.is_running():
            pass
        else:
            # set running indicator to block double readout
            self.running.now_running(self)
            # switch start to stop button
            self.startBtn.label = 'Stop'
            self.startBtn.button_type = 'danger'
            # set timeconstant and integration
            self.integration = float(self.integrationInput.value)

            # create the measurment thread
            self.cont_readout = measurement.measurement(
                inputs=None,
                sequence=[self.spectrometer.frame],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=partial(
                    self.spectrometer.setup, integration=self.integration),
                finish=None,
                save_output=False)
            # start the measurment thread
            self.cont_readout.start()

    def stop(self):
        if not (self.cont_readout is None):
            self.cont_readout.stop()
            self.cont_readout.join()
        self.running.now_stopped()
        self.startBtn.label = 'Start'
        self.startBtn.button_type = 'success'

    def close(self):
        self.stop()

    def save_spectrum(self):
        self.linePlot.save_current(name=self.saveNameInput.value, num=0)

    def update(self, data):
        if time.time() - self.lastrefresh > 0.05:
            if not (data is None):
                w = data[0]['wavelength']
                i = data[0]['intensity']
                self.linePlot.update(num=0, x=w, y=i)
            self.lastrefresh = time.time()


class whitelight_measurement_gui(whitelight_alignment_gui):
    def __init__(self, doc, running, spectrometer):
        super().__init__(doc, running, spectrometer)
        self.title = 'Measurement'

        self.BG = None
        self.A = None
        self.B = None
        self.AB = None
        self.GD = None
        self.gdaxis = None
        self.w = None
        self.FFT = None

        self.currentlyBG = False
        self.currentlyA = False
        self.currentlyB = False
        self.currentlyAB = False
        self.currentlygd = False

        self.replot = True

        self.counter = 0
        self.limitspectra = 100

        # extra line plots
        self.linePlot.line(legend='Background', line_color='grey')
        self.linePlot.line(legend='A', line_color='blue')
        self.linePlot.line(legend='B', line_color='green')
        self.linePlot.line(legend='A+B', line_color='black')

        self.linePlot2 = bgh.plot_2d()
        self.linePlot2.line(legend='GD', line_color='red')
        self.linePlot3 = bgh.plot_2d()
        self.linePlot3.line(legend='FFT', line_color='red')

        self.BGBtn = bokeh.models.widgets.Toggle(label='Background')
        self.ABtn = bokeh.models.widgets.Toggle(label='A')
        self.BBtn = bokeh.models.widgets.Toggle(label='B')
        self.ABBtn = bokeh.models.widgets.Toggle(label='A+B')

        self.saveNameInput = bokeh.models.widgets.TextInput(
            title='Save Filename', value='Name')

        # arrange items
        self.inputs = bokeh.layouts.widgetbox(
            self.startBtn, self.integrationInput,
            self.BGBtn, self.ABtn, self.BBtn, self.ABBtn,
            self.saveBtn, self.saveNameInput)
        self.layout = bokeh.layouts.row(
            self.inputs,
            bokeh.layouts.column(
                self.linePlot.element,
                self.linePlot2.element,
                self.linePlot3.element),
            width=800)

    def save_spectrum(self):
        self.now = datetime.datetime.now()
        name = self.saveNameInput.value
        # save data text
        os.makedirs(
            self.now.strftime('%Y-%m')
            + '/' + self.now.strftime('%Y-%m-%d'),
            exist_ok=True)
        fname = (self.now.strftime('%Y-%m') + '/'
                 + self.now.strftime('%Y-%m-%d')
                 + '/whitelight_'
                 + name + '_'
                 + self.now.strftime('%Y-%m-%d-%H-%M-%S'))

        # save data hdf5
        try:
            with h5py.File(fname + '.hdf5', 'w') as f:
                f.create_dataset('wavelength', data=self.w)
                f.create_dataset('A', data=self.A)
                f.create_dataset('B', data=self.B)
                f.create_dataset('AB', data=self.AB)
                f.create_dataset('background', data=self.BG)
                f.create_dataset('wavelengthGD', data=self.gdaxisnm)
                f.create_dataset('GDnm', data=self.GDnm)
                f.create_dataset('GDPHz', data=self.GD)
                f.create_dataset('frequencyGD', data=self.gdaxis)
                f.flush()
        except Exception as e:
            print('save error')
            print(e)

    def update(self, data):
        if not (data is None):
            w = data[0]['wavelength']
            i = data[0]['intensity']
            self.w = w

            # fix when the arrays are uninitialized
            if self.BG is None:
                self.A = w*0
                self.B = w*0
                self.AB = w*0
                self.BG = w*0
                self.GD = w*0
                self.gdaxis = w*0
                self.FFT = w*0

            if self.BGBtn.active:
                self.BG = (self.BG*self.counter + i) / (self.counter+1.)
                self.counter += 1
                if self.counter >= self.limitspectra:
                    self.BGBtn.active = False
                    self.counter = 0
                    self.replot = True
            if self.ABtn.active:
                self.A = (self.A*self.counter + i) / (self.counter+1.)
                self.counter += 1
                if self.counter >= self.limitspectra:
                    self.ABtn.active = False
                    self.counter = 0
                    self.replot = True
            if self.BBtn.active:
                self.B = (self.B*self.counter + i) / (self.counter+1.)
                self.counter += 1
                if self.counter >= self.limitspectra:
                    self.BBtn.active = False
                    self.counter = 0
                    self.replot = True
            if self.ABBtn.active:
                self.AB = (self.AB*self.counter + i) / (self.counter+1.)
                self.counter += 1
                if self.counter >= self.limitspectra:
                    self.ABBtn.active = False
                    self.counter = 0
                    self.replot = True

            # update plots
            if time.time() - self.lastrefresh > 0.1:
                # update current spectrum
                self.linePlot.update(num=0, x=w, y=i-self.BG)

                # plot fft
                f = 3e8/(w*1e-9)/1e15
                curr = (i-self.BG)/(f**2)
                new_frequency, new_spec = fourierAndAxis.fixPosAxis(f, curr)
                ftSig = fourierAndAxis.positiveSpectrumToTimeDomain(
                    new_frequency, new_spec)
                self.fftaxis = ftSig.timeAxisV
                self.FFT = np.abs(ftSig.signalTimeDomainV)
                self.FFT = self.FFT[(0 < self.fftaxis) & (1500 > self.fftaxis)]
                self.fftaxis = (
                    self.fftaxis[(0 < self.fftaxis) & (1500 > self.fftaxis)])
                self.linePlot3.update(num=0, x=self.fftaxis, y=self.FFT)

                # update the other spectra
                if self.replot:
                    self.replot = False
                    self.linePlot.update(num=1, x=w, y=self.BG)
                    self.linePlot.update(num=2, x=w, y=self.A-self.BG)
                    self.linePlot.update(num=3, x=w, y=self.B-self.BG)
                    self.linePlot.update(num=4, x=w, y=self.AB-self.BG)

                    # calc gd
                    f = 3e8/(w*1e-9)/1e15
                    a = (self.A-self.BG) / (f**2)
                    b = (self.B-self.BG) / (f**2)
                    ab = (self.AB-self.BG) / (f**2)
                    new_frequency, new_spec = fourierAndAxis.fixPosAxis(
                        f, ab-a-b)
                    ftSig = fourierAndAxis.positiveSpectrumToTimeDomain(
                        new_frequency, new_spec)
                    signalTimeDomainVFILTERED = copy.copy(
                        ftSig.signalTimeDomainV)
                    signalTimeDomainVFILTERED[
                        (ftSig.timeAxisV < 500) | (ftSig.timeAxisV > 750)] = 0
                    ftBack = fourierAndAxis.fieldToFrequencyDomain(
                        ftSig.timeAxisV, signalTimeDomainVFILTERED)
                    self.gdaxis = ftBack.gdAxisPosV
                    self.GD = ftBack.gdPosV

                    # plot gd
                    self.gdaxisnm = 3e8/(self.gdaxis*1e-9)/1e15
                    self.GDnm = self.GD[
                        (w.min() < self.gdaxisnm) & (w.max() > self.gdaxisnm)]
                    self.gdaxisnm = self.gdaxisnm[
                        (w.min() < self.gdaxisnm) & (w.max() > self.gdaxisnm)]
                    self.linePlot2.update(num=0, x=self.gdaxisnm, y=self.GDnm)

                self.lastrefresh = time.time()


class lockin_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        # create object to determine if some measurement is currently running
        running = bgh.running()

        # open hardware
        spectrometer = oceanoptics_spectrometer()

        alignmentgui = whitelight_alignment_gui(
            doc=doc, running=running, spectrometer=spectrometer)

        # measurement tab
        measurementgui = whitelight_measurement_gui(
            doc=doc, running=running, spectrometer=spectrometer)

        # documents properties
        self.title = 'Whitelight Readout'
        self.tabs = [
            {'layout': alignmentgui.layout, 'title': alignmentgui.title},
            {'layout': measurementgui.layout, 'title': measurementgui.title}]

        # this list is auto-closed, all close functions of the added objects
        # are called at session destruction
        self.close_list.append(alignmentgui)
        self.close_list.append(measurementgui)
        self.close_list.append(spectrometer)


# start the server
print('start whitelight')
bgh.bokeh_gui_helper(lockin_session_handler(), 5025)
