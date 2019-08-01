import bokeh
from functools import partial

from small_lab_gui.helper import bokeh_gui_helper as bgh
from small_lab_gui.helper import asynczmq
from small_lab_gui.helper import measurement

# for the lab
# from small_lab_gui.axes.linear_axis_pi_c885 import linear_axis_controller_pimicosc885
# from small_lab_gui.axes.linear_axis_pi_c885 import linear_axis_pimicosc885

# for testing
from small_lab_gui.axes.linear_axis_dummy import linear_axis_controller_dummy as linear_axis_controller_pimicosc885
from small_lab_gui.axes.linear_axis_dummy import linear_axis_dummy as linear_axis_pimicosc885


class axis_gui():
    # creator
    def __init__(self, doc, name, menu, axis):
        # running indicator
        self.running = bgh.running()

        # axis name and positions menu
        self.doc = doc
        self.name = name
        self.menu = menu

        self.axis = axis

        # widget items
        self.titleBar = bokeh.models.widgets.Paragraph(
            text=self.name, width=180)
        self.positionsDropdown = bokeh.models.widgets.Dropdown(
            label=str(self.axis.position), button_type='primary',
            menu=self.menu, width=180)
        self.distanceInput = bokeh.models.widgets.TextInput(
            title='Distance [mm]', value='1.0', width=170)
        self.signSwitchToggle = bokeh.models.widgets.Toggle(
            label='Positive', button_type='success', width=180)
        self.startBtn = bokeh.models.widgets.Button(
            label='Start ', button_type='primary', width=180)
        self.inputs = bokeh.layouts.widgetbox(
            self.titleBar, self.positionsDropdown, self.distanceInput,
            self.signSwitchToggle, self.startBtn, width=200)

        self.homeBtn = bokeh.models.widgets.Button(
            label='Home ' + self.name, button_type='primary', width=180)
        self.zeroBtn = bokeh.models.widgets.Button(
            label='Zero ' + self.name, button_type='primary', width=180)

        if self.axis.referenced:
            self.startBtn.label = 'Start'
            self.startBtn.disabled = False
        else:
            self.startBtn.label = 'Reference first!'
            self.startBtn.disabled = True

        # callbacks
        self.signSwitchToggle.on_change('active', self.signSwitch_change)
        self.positionsDropdown.on_change('value', self.dropdown_change)
        self.startBtn.on_click(self.start_movement_rel)
        self.homeBtn.on_click(self.home)
        self.zeroBtn.on_click(self.zero)

        # check if at defined position
        for entry in self.menu:
            if self.axis.position == float(entry[1]):
                self.positionsDropdown.label = entry[0]

    # handle a sign button press
    def signSwitch_change(self, attr, old, new):
        if self.signSwitchToggle.button_type == 'success':
            self.signSwitchToggle.button_type = 'danger'
            self.signSwitchToggle.label = 'negative'
        else:
            self.signSwitchToggle.button_type = 'success'
            self.signSwitchToggle.label = 'positive'

    # home axis
    def home(self):
        self.start_movement(self.axis.home, btn=self.homeBtn)

    # zero axis
    def zero(self):
        self.start_movement(self.axis.zero, btn=self.zeroBtn)

    # perform an absolute move
    def start_movement_abs(self, aim_position):
        if self.axis.referenced:
            self.start_movement(
                partial(
                    self.axis.abs_move, aim_position=aim_position),
                btn=self.startBtn)

    # perform a relative move
    def start_movement_rel(self):
        # get abs position
        currentPosition = self.axis.position
        # determine direction
        if self.signSwitchToggle.button_type == 'success':
            sign = 1
        else:
            sign = -1
        # calculate distance
        relchange = sign * float(self.distanceInput.value)
        # use absolute move function
        self.start_movement_abs(aim_position=currentPosition+relchange)

    # perform an absolute move requested by the dropdown
    def dropdown_change(self, attr, old, new):
        self.start_movement_abs(aim_position=float(new))

    def start_movement_abs_remote(self, aim_position):
        # start the movement
        self.start_movement_abs(aim_position=aim_position)
        # wait for it to finish
        self.movement.join()

    # perform a relative move
    def start_movement_rel_remote(self, relchange=0):
        # get abs position
        currentPosition = self.axis.position
        # use absolute move function
        self.start_movement_abs(aim_position=currentPosition+relchange)
        # wait for it to finish
        self.movement.join()

    # perform some abstract movement
    def start_movement(self, call, btn, aim_position=0):
        # in case this measurement is running, stop it
        if self.running.am_i_running(self):
            self.stop()
        # in case a different measurement is running, do nothing
        elif self.running.is_running():
            pass
        else:
            # set running indicator to block double readout
            self.running.now_running(self)

            self.movement = measurement.measurement(
                inputs=None,
                sequence=[
                    measurement.sleep_function(0.05),
                    self.axis.follow_move],
                update=measurement.bokeh_update_function(
                    self.update, self.doc),
                init=call,
                finish=measurement.bokeh_no_input_finish_function(
                    self.restore, self.doc),
                save_output=False)

            # change button to stop if requested
            btn.label = 'stop'
            btn.button_type = 'danger'

            # start movement
            self.movement.start()

    def stop(self):
        self.axis.stop()
        if self.movement is not None:
            # measurement is inherited from Thread,
            # so one can wait for it using join
            self.movement.join()
        self.running.now_stopped()
        self.startBtn.label = 'Start'
        self.startBtn.button_type = 'success'
        # stop movement here

    # update position
    def update(self, data):
        position = data[1]
        for entry in self.menu:
            if position == float(entry[1]):
                self.positionsDropdown.label = entry[0]
                self.positionsDropdown.value = ''
                break
        else:
            self.positionsDropdown.label = str(position) + ' mm'

    # restore some button to before movement state
    def restore(self):
        if self.axis.referenced:
            self.startBtn.label = 'Start'
            self.startBtn.disabled = False
        else:
            self.startBtn.label = 'Reference first!'
            self.startBtn.disabled = True

        self.homeBtn.label = 'Home ' + self.name
        self.zeroBtn.label = 'Zero ' + self.name

        self.startBtn.button_type = 'primary'
        self.homeBtn.button_type = 'primary'
        self.zeroBtn.button_type = 'primary'

        position = self.axis.get_position()
        for entry in self.menu:
            if position == float(entry[1]):
                self.positionsDropdown.label = entry[0]
                self.positionsDropdown.value = '1e100'
                break
        else:
            self.positionsDropdown.label = str(position) + ' mm'

        self.running.now_stopped()


class movement_session_handler(bgh.bokeh_gui_session_handler):
    def open_session(self, doc):
        all_controllers = []
        all_axes = []
        all_axes_guis = []
        homeBtns = []
        zeroBtns = []

        # provide functionality to other scripts
        waiter = asynczmq.asyncZmqHandler(bokeh_doc=doc)

        # controller
        move_contr = linear_axis_controller_pimicosc885()

        # hhg_target_stage
        axis = linear_axis_pimicosc885(move_contr, 0)
        ax_gui = axis_gui(
            doc=doc,
            name='HHG Target Focus',
            menu=[('Focus', '37.5'), ],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage(('hhg_abs_move', ax_gui.start_movement_abs_remote))
        waiter.addMessage(('hhg_get_position', axis.get_position))

        # filter_wheel
        axis = linear_axis_pimicosc885(move_contr, 1)
        ax_gui = axis_gui(
            doc=doc,
            name='Filter Wheel',
            menu=[
                ('0.5um Al', '0.97'),
                ('0.5um In', '1.32'),
                ('No Filter', '0.62'),
                ('0.1um In', '0.27'),
                ('25.0um Glass', '1.67'),
                ('0.2um In', '2.02'), ],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage(
            ('filterwheel_abs_move', ax_gui.start_movement_abs_remote))
        waiter.addMessage(
            ('filterwheel_get_position', axis.get_position))

        # experimental_mirror_stage
        axis = linear_axis_pimicosc885(move_contr, 3)
        ax_gui = axis_gui(
            doc=doc,
            name='Experimental Mirror',
            menu=[
                ('Out', '0'),
                ('Profile', '49'),
                ('Grating', '124')],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage(
            ('experimental_mirror_abs_move', ax_gui.start_movement_abs_remote))
        waiter.addMessage(
            ('experimental_mirror_get_position', axis.get_position))

        # target_stage_z
        axis = linear_axis_pimicosc885(move_contr, 4)
        ax_gui = axis_gui(
            doc=doc,
            name='Experimental Target Z',
            menu=[
                ('Current', '68.2'),
                ('Nozzle', '35.9')],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage((
            'experimental_target_z_abs_move',
            ax_gui.start_movement_abs_remote))
        waiter.addMessage(
            ('experimental_target_z_get_position', axis.get_position))

        # target_stage_x
        axis = linear_axis_pimicosc885(move_contr, 5)
        ax_gui = axis_gui(
            doc=doc,
            name='Experimental Target X',
            menu=[
                ('Out', '0'),
                ('Current', '22.252'),
                ('Nozzle', '40.85')],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage((
            'experimental_target_x_abs_move',
            ax_gui.start_movement_abs_remote))
        waiter.addMessage(
            ('experimental_target_x_get_position', axis.get_position))

        # piezo
        """#axis = linear_axis.linear_axis_pollux(move_contr, '3')
        axis = linear_axis.linear_axis_pimicosc885(move_contr, 8)
        #axis = linear_axis.linear_axis_dummy()
        ax_gui = axis_gui(
            doc=doc,
            name='Piezo',
            menu=[
                ('Away', '0'),
                ('Overlap', '10')],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage(('piezo_abs_move', ax_gui.start_movement_abs_remote))
        waiter.addMessage(('piezo_get_position', axis.get_position)) """

        # Delay XUV
        axis = linear_axis_pimicosc885(move_contr, 2)
        ax_gui = axis_gui(
            doc=doc,
            name='Delay XUV',
            menu=[
                ('XUV Profile', '8'),
                ('IR Profile', '95'),
                ('Transmition', '150')],
            axis=axis)
        all_axes.append(axis)
        all_axes_guis.append(ax_gui.inputs)
        homeBtns.append(ax_gui.homeBtn)
        zeroBtns.append(ax_gui.zeroBtn)

        waiter.addMessage(
            ('Delay_XUV_abs_move', ax_gui.start_movement_abs_remote))
        waiter.addMessage(('Delay_XUV_position', axis.get_position))

        # movement tab
        controlColumn = bokeh.layouts.column(bokeh.layouts.row(all_axes_guis))
        # home button tab
        homeColumn = bokeh.layouts.column(bokeh.layouts.widgetbox(homeBtns))
        zeroColumn = bokeh.layouts.column(bokeh.layouts.widgetbox(zeroBtns))
        advancedRow = bokeh.layouts.row(homeColumn, zeroColumn)

        # start the message server
        waiter.start()

        self.title = 'Mover'
        self.tabs = [
            {'layout': controlColumn, 'title': 'Movement'},
            {'layout': advancedRow, 'title': 'Home and Zero'}]

        # this list is auto-closed, all close function of the
        # added objects are called at session destruction
        self.close_list.append(waiter)
        for axis in all_axes:
            self.close_list.append(axis)
        for controller in all_controllers:
            self.close_list.append(controller)


print('start movement')
# start the server
bgh.bokeh_gui_helper(movement_session_handler(), 5007)
