import numpy as np
import matplotlib.pyplot as plt


class fieldToFrequencyDomain:
    """calculates the frequency domain representation of an electric field,
    given in the time domain.
    if time is input in seconds, frequency axis is in Hertz
    if time is input in milliseconds, frequency axis is in kHz ....
    normalization keeps the squared integral constant"""
    def __init__(self, timeAxisV, signalTimeDomainV):
        # save input
        self.timeAxisV = np.array(timeAxisV)
        self.signalTimeDomainV = np.array(signalTimeDomainV)

        # frequency axis and spacing
        self.dtS = self.timeAxisV[1]-self.timeAxisV[0]
        self.frequencyAxisV = np.fft.fftfreq(self.timeAxisV.size, d=self.dtS)
        self.dfS = self.frequencyAxisV[1]-self.frequencyAxisV[0]

        # do the fourier transform and keep power constant
        self.signalFrequencyDomainV = (
            np.fft.fft(self.signalTimeDomainV)*self.dtS)

        # spectrum and phase
        self.signalFrequencyDomainAbsV = np.abs(self.signalFrequencyDomainV)
        self.signalFrequencyDomainPhaseV = np.angle(
            self.signalFrequencyDomainV)

        # calculate group delay
        # fourier window is actually one step bigger than expected
        # (space to the next window on each side)
        self.timeRangeS = (np.max(self.timeAxisV)
                           - np.min(self.timeAxisV) + self.dtS)
        # gd must be in the window length (this will push the minimum to zero)
        self.gdV = (-1.*np.diff(self.signalFrequencyDomainPhaseV)
                    / np.diff(self.frequencyAxisV*2.*np.pi))
        # minimum should actually be the minimum of the fourier window
        self.gdV = np.mod(self.gdV, self.timeRangeS) + np.min(self.timeAxisV)

        self.gdAxisV = (self.frequencyAxisV[1:]+self.frequencyAxisV[0:-1])/2.
        self.gdAxisPosV = self.gdAxisV[self.gdAxisV >= 0.]

        # unwrap the phase
        self.signalFrequencyDomainPhaseV = np.concatenate(
            ([self.signalFrequencyDomainPhaseV[0]],
             self.signalFrequencyDomainPhaseV[0]
             + np.cumsum(self.gdV)*self.dfS*2*np.pi))

        # isolate positive frequencies (contain all information for real input)
        self.signalFrequencyDomainPosV = (
            np.sqrt(2.)*self.signalFrequencyDomainV[self.frequencyAxisV >= 0.])
        self.frequencyAxisPosV = self.frequencyAxisV[self.frequencyAxisV >= 0.]
        self.signalFrequencyDomainAbsPosV = (
            self.signalFrequencyDomainAbsV[self.frequencyAxisV >= 0.])
        self.signalFrequencyDomainPhasePosV = (
            self.signalFrequencyDomainPhaseV[self.frequencyAxisV >= 0.])
        self.gdPosV = self.gdV[self.gdAxisV >= 0.]

    # functions to avoid intermediate variables
    def echoSpectrumPos(self):
        return self.frequencyAxisPosV, self.signalFrequencyDomainPosV

    def echoSpectrum(self):
        return self.frequencyAxisV, self.signalFrequencyDomainV


class getEnvelope(fieldToFrequencyDomain):
    """calculates the envelope of an oscillating signal by subtracting the
    carrier frequency, which can be given or will be determined"""
    def __init__(self, timeAxisV, signalTimeDomainV, carrierFrequencyS=None):
        # get spectral domain data
        super().__init__(timeAxisV, signalTimeDomainV)
        # eliminate negative frequencies
        self.signalFrequencyDomainV[self.frequencyAxisV < 0] = 0
        # if carrier is requested, use it, otherwise find best one
        if carrierFrequencyS:
            self.carrierFrequencyS = carrierFrequencyS
        else:
            self.carrierFrequencyS = (
                np.trapz(
                    self.signalFrequencyDomainAbsPosV**2
                    * self.frequencyAxisPosV,
                    self.frequencyAxisPosV)
                / np.trapz(
                    self.signalFrequencyDomainAbsPosV**2,
                    self.frequencyAxisPosV))
        # put the carrier as close as possible to zero frequency
        self.signalFrequencyDomainV = (
            2*(np.roll(
                self.signalFrequencyDomainV,
                -np.argmin(np.abs(self.frequencyAxisV
                                  - self.carrierFrequencyS)))))
        # get the time domain result
        backTransform = fullSpectrumToTimeDomain(
            self.frequencyAxisV,
            self.signalFrequencyDomainV,
            np.min(self.timeAxisV))
        self.timeAxisV = backTransform.timeAxisV
        self.envelopeV = np.abs(backTransform.signalTimeDomainComplexV)
        self.envelopePhaseV = np.angle(backTransform.signalTimeDomainComplexV)

    # functions to avoid intermediate variables
    def echoEnvelope(self):
        return self.timeAxisV, self.envelopeV

    def echoPhase(self):
        return self.timeAxisV, self.envelopePhaseV

    def echoEnvelopePhase(self):
        return self.timeAxisV, self.envelopeV, self.envelopePhaseV


class fullSpectrumToTimeDomain:
    """calculates the time domain representation of an electric field,
    given in the frequency domain including negative frequencies.
    if time is input in seconds, frequency axis is in Hertz
    if time is input in milliseconds, frequency axis is in kHz ....
    normalization keeps the squared integral constant"""
    def __init__(self, frequencyAxisV, signalFrequencyDomainV, timeZeroS=0.):
        self.frequencyAxisV = np.array(frequencyAxisV)
        self.signalFrequencyDomainV = np.array(signalFrequencyDomainV)
        # single sided spectrum for completeness
        self.signalFrequencyDomainPosV = (
            np.sqrt(2.)*self.signalFrequencyDomainV[self.frequencyAxisV >= 0.])
        self.frequencyAxisPosV = self.frequencyAxisV[self.frequencyAxisV >= 0.]

        self.dfS = self.frequencyAxisV[1]-self.frequencyAxisV[0]

        self.timeAxisV = (
            np.linspace(0., 1./self.dfS,
                        len(self.frequencyAxisV), endpoint=False)
            + timeZeroS)
        self.dtS = self.timeAxisV[1]-self.timeAxisV[0]

        self.signalTimeDomainComplexV = np.fft.ifft(
            self.signalFrequencyDomainV)/self.dtS
        self.signalTimeDomainV = np.real(self.signalTimeDomainComplexV)

    # functions to avoid intermediate variables
    def echoSignalComplex(self):
        return self.timeAxisV, self.signalTimeDomainComplexV

    def echoSignalReal(self):
        return self.timeAxisV, self.signalTimeDomainV


class positiveSpectrumToTimeDomain(fullSpectrumToTimeDomain):
    """calculates the time domain representation of an electric field,
    given in the frequency domain only including positive frequencies.
    if time is input in seconds, frequency axis is in Hertz
    if time is input in milliseconds, frequency axis is in kHz ....
    normalization keeps the squared integral constant"""
    def __init__(self, frequencyAxisPosV,
                 signalFrequencyDomainPosV, timeZeroS=0.):
        # make sure input are numpy arrays
        frequencyAxisPosV = np.array(frequencyAxisPosV)
        signalFrequencyDomainPosV = np.array(signalFrequencyDomainPosV)

        # the inserted frequency is the highest frequency, which is lost
        # because it is only present in the negative frequency spectrum
        # (positive frequencies have the zero,
        # which is not present in the negative frequencies)
        frequencyAxisV = np.concatenate(
            (frequencyAxisPosV,
             [-2*frequencyAxisPosV[-1]+frequencyAxisPosV[-2]],
             -frequencyAxisPosV[-1:0:-1]))
        signalFrequencyDomainV = np.concatenate(
            (signalFrequencyDomainPosV,
             [0],
             np.conj(signalFrequencyDomainPosV[-1:0:-1])))/np.sqrt(2.)

        # once full spectra are generated, use existing code
        super().__init__(frequencyAxisV, signalFrequencyDomainV, timeZeroS)


def fixPosAxis(frequencyV, signalV, overSampling=10):
    """fixes a signal that is unequally spaced or does not start at freq=0
    e.g. if you want to fft a spectrometer signal"""
    freq = np.array(frequencyV)
    signal = np.array(signalV)
    if freq[1] < freq[0]:
        freq = np.flip(freq)
        signal = np.flip(signal)
    df = np.min(np.diff(freq))/overSampling
    newFreq = np.arange(0, np.max(freq), df)
    newSignal = np.interp(newFreq, freq, signal, left=0, right=0)
    return newFreq, newSignal
