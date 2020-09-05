from threading import Thread, Event
from functools import partial
import time
import inspect


class measurement(Thread):
    """class that performs a measurement sequence in a new thread."""
    def __init__(self, inputs=None, sequence=None, init=None,
                 update=None, finish=None, save_output=True):
        """saves all measurement parameters. does not run the measurement.

        Keyword arguments:
        inputs -- list of lists containing the input parameter for the
            sequence functions . if not passed, an infinite measurement
            is performed with no input
        sequence -- list of sequence functions call in order
        init -- function called before sequence, i.e. setup
        update -- function called after each successfull step
        finish -- function called after all steps. will be called even in
            case of exception
        save_output -- determines if the sequences' output is collected or
            discarded after each step. accordingly, update will receive a list
            with the output of the current step only, or a list of lists
            containing all past outputs
        """
        super().__init__()
        # inputs for a finite measurement
        self.inputs = inputs
        # all repeated measurement step callbacks
        self.sequence = sequence
        # output list
        self.output = []
        # init function (called once)
        self.init = init
        # update callback (called after every step)
        self.update = update
        # finish function (called once)
        self.finish = finish
        # bool if accumulating output or just treating every single step
        self.save_output = save_output
        # container for the init function output
        self.init_output = None
        # container for the finish function output
        self.finish_output = None
        # stop event to stop measurement prematurely
        self.stop_event = Event()

    def stop(self):
        """stops the measurement before executing the next step and if
        supported by the sequence functions also the currently running step"""
        self.stop_event.set()

    def run(self):
        """performs the measurement. do not call directly, use start()"""
        # this list holds all output of steps and sub-steps
        self.output = []
        # call the init function
        try:
            if self.init:
                self.init_output = self.init()
        except Exception as e:
            print('error during measurement init')
            print(e)
        else:
            try:
                # if discrete set of variables is given, do finite measurement
                if self.inputs is not None:
                    # every entry in inputs is a step
                    for inplist in self.inputs:
                        # break condition
                        if self.stop_event.is_set():
                            break
                        # this list holds all sub step results in a step
                        retlist = []
                        # every entry in sequence is a sub step
                        for step, inp in zip(self.sequence, inplist):
                            # break condition
                            if self.stop_event.is_set():
                                break
                            # execute sub step and save result
                            retlist.append(
                                step(self.stop_event, inp, self.init_output))
                        # if the whole step executed,
                        # save the step output as list and call update routine
                        if not self.stop_event.is_set():
                            if self.save_output:
                                self.output.append(retlist)
                                if self.update:
                                    self.update(self.output.copy())
                            else:
                                if self.update:
                                    self.update(retlist.copy())
                # otherwise do an infinite measurement
                else:
                    # break condition
                    while not self.stop_event.is_set():
                        # this list holds all sub step results in a step
                        retlist = []
                        # every entry in sequence is a sub step
                        for step in self.sequence:
                            # execute sub step and save result
                            retlist.append(
                                step(self.stop_event, None, self.init_output))
                        # if the whole step executed, save the step output
                        #  as list and call the update routine
                        if not self.stop_event.is_set():
                            if self.save_output:
                                self.output.append(retlist)
                                if self.update:
                                    self.update(self.output.copy())
                            else:
                                if self.update:
                                    self.update(retlist)
            except Exception as e:
                print('error during measurement sequence')
                print(e)
        finally:
            try:
                # call the finishing routine
                if self.finish:
                    self.finish_output = self.finish(
                        self.output, self.init_output)
            except Exception as e:
                print('error during measurement finish')
                print(e)
                return
        print('measurement thread ended')
        return(0)


def no_input_sequence_function(function):
    """return a function that is compatible with measurement class from a
    function with no input (no stop event, no init input)."""
    return lambda stop_event, inp, init_output: function()


def single_input_sequence_function(function):
    """return a function that is compatible with measurement class from a
    function with a single input (no stop event, no init input)."""
    return lambda stop_event, inp, init_output: function(inp)


def sleep_function(sleeptime_sec):
    """return a function that is compatible with measurement class
    and waits sleeptime_sec."""
    return lambda stop_event, inp, init_output: time.sleep(sleeptime_sec)


def bokeh_update_function(update_function, doc):
    """return a function that adds a callback to the bokeh server queue."""
    if inspect.iscoroutinefunction(update_function):
        return lambda data: doc.add_next_tick_callback(
            partial(update_function, data=data))
    else:
        async def async_update_function(data):
            update_function(data)
        return lambda data: doc.add_next_tick_callback(
            partial(async_update_function, data=data))


def bokeh_no_input_update_function(update_function, doc):
    """return a function that adds a callback to the bokeh server queue."""
    if inspect.iscoroutinefunction(update_function):
        return lambda data: doc.add_next_tick_callback(update_function)
    else:
        async def async_update_function():
            update_function()
        return lambda data: doc.add_next_tick_callback(async_update_function)


def bokeh_finish_function(update_function, doc):
    """return a finish function that adds a callback
    to the bokeh server queue."""
    if inspect.iscoroutinefunction(update_function):
        return lambda output, init_output: doc.add_next_tick_callback(
            partial(update_function, output, init_output))
    else:
        async def async_update_function(output, init_output):
            update_function(output, init_output)
        return lambda output, init_output: doc.add_next_tick_callback(
            partial(async_update_function, output, init_output))


def bokeh_no_input_finish_function(update_function, doc):
    """return a finish function that adds a callback to the bokeh server queue
    from a function without inputs."""
    if inspect.iscoroutinefunction(update_function):
        return lambda output, init_output: doc.add_next_tick_callback(
            update_function)
    else:
        async def async_update_function():
            update_function()
        return lambda output, init_output: doc.add_next_tick_callback(
            async_update_function)
