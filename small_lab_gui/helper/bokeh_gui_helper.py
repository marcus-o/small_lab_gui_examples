import bokeh
import bokeh.layouts
import bokeh.plotting

import bokeh.server.server
import bokeh.application
import bokeh.application.handlers.function

import socket
import time
import numpy as np
from numba import jit


class running():
    def __init__(self):
        self.running = False
        self.who = None

    def now_running(self, who=None):
        if not self.running:
            self.running = True
            self.who = who
        else:
            raise Exception('Was already running, this should not happen')

    def now_stopped(self):
        self.running = False

    def is_running(self):
        return self.running

    def am_i_running(self, who):
        if who is self.who:
            return self.running
        else:
            return False


class bokeh_gui_session_handler(bokeh.application.handlers.Handler):
    def modify_document(self, doc):
        print('opening session')
        # standard values
        self.close_list = []
        self.title = ''
        self.tabs = []
        self.open_session(doc)

        # bokeh stuff
        doc.add_root(
            bokeh.models.widgets.Tabs(
                tabs=[bokeh.models.widgets.Panel(
                    child=c['layout'], title=c['title']) for c in self.tabs]
            ))
        doc.title = self.title

    def on_session_destroyed(self, doc):
        print('closing session')
        self.close_stuff()
        self.destroy_session(doc)
        if self.server is not None:
            self.server.io_loop.stop()

    def open_session(self, doc):
        pass

    def destroy_session(self, doc):
        pass

    def close_stuff(self):
        for obj in self.close_list:
            try:
                obj.close()
            except Exception as e:
                print('close error')
                print(e)

    def stop_server_after_session(self, server):
        self.server = server


class bokeh_gui_helper():
    def __init__(self, session_handler, min_port):
        self.handler = session_handler

        self.bokeh_port = try_port(port=min_port, pmax=min_port+100)
        if self.bokeh_port is not None:
            self.apps = {
                '/': bokeh.application.application.Application(self.handler)}
            self.server = bokeh.server.server.Server(
                self.apps,
                port=self.bokeh_port,
                unused_session_lifetime_milliseconds=5000,
                check_unused_sessions_milliseconds=1000
                )
            print('server ready')
            self.handler.stop_server_after_session(self.server)
            self.server.start()
            self.server.show('/')
            self.server.run_until_shutdown()
        else:
            print('no free port')


# function that returns the first free port in a range
def try_port(port=5000, pmax=5100):
    success = False
    while (port <= pmax) and not success:
        try:
            serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serversocket.bind((socket.gethostname(), port))
            success = True
        except Exception:
            port = port+1
        finally:
            serversocket.close()
    if success:
        return port
    else:
        return None


# 2d plot helper class
class plot():
    def __init__(self, title='', height=256, width=1024, max_refresh=0.05):
        self.element = self.plot = bokeh.plotting.figure(
            plot_height=height, plot_width=width,
            title=title,
            tools='crosshair,reset,save,box_zoom,box_select,pan,wheel_zoom'
            )
        self.max_refresh = max_refresh
        self.lastrefresh = []

    def update(self):
        pass

    def set_bounds(self, xmax=None, xmin=None, ymax=None, ymin=None,
                   xrange=None, yrange=None):
        if xmax is not None:
            self.plot.x_range.end = float(xmax)
        if xmin is not None:
            self.plot.x_range.start = float(xmin)
        if ymax is not None:
            self.plot.y_range.end = float(ymax)
        if ymin is not None:
            self.plot.y_range.start = float(ymin)
        if xrange is not None:
            self.plot.x_range.end = float(xrange.max())
            self.plot.x_range.start = float(xrange.min())
        if yrange is not None:
            self.plot.y_range.end = float(yrange.max())
            self.plot.y_range.start = float(yrange.min())


# 2d data source helper class
class source_2d(plot):
    def __init__(self, x=[], y=[]):
        self.source = bokeh.models.ColumnDataSource({'x': x, 'y': y})
        self.x = x
        self.y = y

    def update(self, x, y):
        self.source.data = {'x': x, 'y': y}
        self.x = x
        self.y = y


# 2d plot helper class
class plot_2d(plot):
    def __init__(self, title='', height=256, width=1024, max_refresh=0.05):
        # plot
        self.height = height
        self.width = width
        self.max_refresh = max_refresh
        self.lastrefresh = []

        self.plot = bokeh.plotting.figure(
            plot_height=height, plot_width=width,
            title=title,
            tools='crosshair,reset,save,box_zoom,box_select,pan,wheel_zoom'
            )
        self.plot.x_range = bokeh.models.Range1d(-1e9, 1e9)
        self.plot.y_range = bokeh.models.Range1d(-1e9, 1e9)
        # autoscale buttons
        self.autoscaleXBtn = bokeh.models.Toggle(
            label='Autoscale X', active=True, width=int(np.floor(width/6)))
        self.autoscaleYBtn = bokeh.models.Toggle(
            label='Autoscale Y', active=True, width=int(np.floor(width/6)))
        # replot button
        self.replotBtn = bokeh.models.Button(
            label='Replot', width=int(np.floor(width/6)))
        self.replotBtn.on_click(lambda inp1: self.update_all())

        self.element = bokeh.layouts.column(
            self.plot,
            bokeh.layouts.row(
                bokeh.layouts.widgetbox(self.autoscaleXBtn),
                bokeh.layouts.widgetbox(self.autoscaleYBtn),
                bokeh.layouts.widgetbox(self.replotBtn)))

        self.sources = []
        self.lines = []
        self.data_x = []
        self.data_y = []

    def line(self, legend='', line_color='blue', x=[], y=[]):
        # create source and plot
        source = source_2d(x, y)
        line = self.plot.step(
            x='x', y='y', source=source.source,
            legend=legend, line_color=line_color, mode="center")
        # save plot and source
        self.sources.append(source)
        self.lines.append(line)
        self.lastrefresh.append(0)
        # legend policy
        self.plot.legend.location = 'top_left'
        self.plot.legend.click_policy = 'hide'

        self.data_x.append([])
        self.data_y.append([])

    def save_current(self, name, num=0):
        # create source and plot
        cnt = len(self.plot.select(
            type=bokeh.models.renderers.GlyphRenderer))
        self.plot.line(
            x=self.data_x[num],
            y=self.data_y[num],
            level='underlay',
            line_color=bokeh.palettes.Category10[10][cnt % 10],
            legend=dict(value=name))

    def autoscale(self, checkBtn=True):
        if self.autoscaleXBtn.active or (not checkBtn):
            xmax = -np.inf
            xmin = np.inf
            for elem in self.data_x:
                if len(elem):
                    if np.max(elem) > xmax:
                        xmax = np.max(elem)
                    if np.min(elem) < xmin:
                        xmin = np.min(elem)
            self.set_bounds(
                xmax=xmax,  # + (xmax-xmin)/10,
                xmin=xmin)  # - (xmax-xmin)/10)
        if self.autoscaleYBtn.active or (not checkBtn):
            ymax = -np.inf
            ymin = np.inf
            for elem in self.data_y:
                if len(elem):
                    if np.max(elem) > ymax:
                        ymax = np.max(elem)
                    if np.min(elem) < ymin:
                        ymin = np.min(elem)
            self.set_bounds(
                ymax=ymax + (ymax-ymin)/10,
                ymin=ymin - (ymax-ymin)/10)

    def update_all(self):
        for cnt, _ in enumerate(self.lines):
            self.update(num=cnt)

    def update(self, x=None, y=None, num=0,
               max_length_x=None, max_length_y=None):
        # load old data if no new data is sent
        if x is None:
            x = self.data_x[num]
        if y is None:
            y = self.data_y[num]
        # check if everything is a numpy array
        x = np.array(x)
        y = np.array(y)
        # check if the z matrix is ordered correctly
        if not (x.shape[0] == y.shape[0]):
            print('cannot update 2d source due to shape error')
            return

        # save the full data
        self.data_x[num] = x
        self.data_y[num] = y

        if time.time() - self.lastrefresh[num] > self.max_refresh:
            # handle autoscaling before cutting
            self.autoscale()

            if max_length_x is None:
                max_length_x = self.width
            else:
                max_length_x = np.min([self.width, max_length_x])

            x, y = reduce_x_1d(
                x, y, max_length_x,
                self.plot.x_range.start, self.plot.x_range.end)

            self.sources[num].update(x, y)
            self.lastrefresh[num] = time.time()


# 3d data source helper class
class source_3d():
    def __init__(self, x=None, y=None, z=None):
        if (x is not None) and (y is not None) and (z is not None):
            self.source = bokeh.models.ColumnDataSource({
                'z': [z],
                'x': [np.min(x)],
                'y': [np.min(y)],
                'dw': [np.max(x)-np.min(x)],
                'dh': [np.max(y)-np.min(y)]})
            self.x = x
            self.y = y
            self.z = z
        else:
            self.source = bokeh.models.ColumnDataSource({
                'x': [], 'y': [], 'z': [], 'dw': [], 'dh': []})
            self.x = []
            self.y = []
            self.z = []

    def update(self, x, y, z):
        self.source.data = {
            'z': [z],
            'x': [np.min(x)],
            'y': [np.min(y)],
            'dw': [np.max(x)-np.min(x)],
            'dh': [np.max(y)-np.min(y)]}
        self.x = x
        self.y = y
        self.z = z


# false color plot helper class
class plot_false_color(plot_2d):
    def __init__(self, title='', height=256, width=1024, max_refresh=0.2):
        # plot
        self.height = height
        self.width = width
        self.max_refresh = max_refresh
        self.lastrefresh = []

        self.plot = bokeh.plotting.figure(
            plot_height=height, plot_width=width, title=title,
            tools='crosshair,reset,save,box_zoom,pan,wheel_zoom')
        self.plot.x_range = bokeh.models.Range1d(-1e9, 1e9)
        self.plot.y_range = bokeh.models.Range1d(-1e9, 1e9)
        # color bar
        self.color_mapper = bokeh.models.LinearColorMapper(
            palette=bokeh.palettes.Spectral11, low=1, high=1e7)
        self.color_bar = bokeh.models.ColorBar(
            color_mapper=self.color_mapper, label_standoff=12,
            border_line_color=None, location=(0, 0),
            ticker=bokeh.models.BasicTicker(desired_num_ticks=10))
        self.plot.add_layout(self.color_bar, 'right')
        # manual slider for color bar
        self.colorscaleSilder = bokeh.models.widgets.RangeSlider(
            title='Colorscale', start=0, end=1000, value=(0, 1000),
            width=int(np.floor(width/6)))
        # autoscale buttons
        self.autoscaleXBtn = bokeh.models.Toggle(
            label='Autoscale X', active=True, width=int(np.floor(width/6)))
        self.autoscaleYBtn = bokeh.models.Toggle(
            label='Autoscale Y', active=True, width=int(np.floor(width/6)))
        self.autoscaleZBtn = bokeh.models.Toggle(
            label='Autoscale Z', active=True, width=int(np.floor(width/6)))
        # replot button
        self.replotBtn = bokeh.models.Button(
            label='Replot', width=int(np.floor(width/6)))
        self.replotBtn.on_click(lambda inp1: self.update_all())

        self.element = bokeh.layouts.column(
            self.plot,
            bokeh.layouts.row(
                bokeh.layouts.widgetbox(self.colorscaleSilder),
                bokeh.layouts.widgetbox(self.autoscaleXBtn),
                bokeh.layouts.widgetbox(self.autoscaleYBtn),
                bokeh.layouts.widgetbox(self.autoscaleZBtn),
                bokeh.layouts.widgetbox(self.replotBtn)))

        self.sources = []
        self.images = []
        self.data_x = []
        self.data_y = []
        self.data_z = []

    def image(self, x=None, y=None, z=None):
        # create source and plot
        source = source_3d(x, y, z)
        image = self.plot.image(
            image='z', source=source.source,
            x='x', y='y', dw='dw', dh='dh',
            color_mapper=self.color_mapper)

        # save plot and source
        self.sources.append(source)
        self.images.append(image)
        self.lastrefresh.append(0)
        self.data_x.append([])
        self.data_y.append([])
        self.data_z.append([])

    def autoscale(self, checkBtn=True):
        # x
        if self.autoscaleXBtn.active or (not checkBtn):
            xmax = -np.inf
            xmin = np.inf
            for elem in self.data_x:
                if len(elem):
                    if np.max(elem) > xmax:
                        xmax = np.max(elem)
                    if np.min(elem) < xmin:
                        xmin = np.min(elem)
            self.set_bounds(xmax=xmax, xmin=xmin)
        # y
        if self.autoscaleYBtn.active or (not checkBtn):
            ymax = -np.inf
            ymin = np.inf
            for elem in self.data_y:
                if len(elem):
                    if np.max(elem) > ymax:
                        ymax = np.max(elem)
                    if np.min(elem) < ymin:
                        ymin = np.min(elem)
            self.set_bounds(ymax=ymax, ymin=ymin)
        # z
        zmax = -np.inf
        zmin = np.inf
        for elem in self.data_z:
            if len(elem):
                if np.max(np.max(elem)) > zmax:
                    zmax = np.max(np.max(elem))
                if np.min(np.min(elem)) < zmin:
                    zmin = np.min(np.min(elem))

        # update range slider
        self.colorscaleSilder.start = zmin
        self.colorscaleSilder.end = zmax
        self.colorscaleSilder.step = (zmax - zmin)/1000
        # update color scale
        if self.autoscaleZBtn.active or (not checkBtn):
            self.color_mapper.low = zmin
            self.color_mapper.high = zmax
        else:
            self.color_mapper.low = self.colorscaleSilder.value[0]
            self.color_mapper.high = self.colorscaleSilder.value[1]

    def update(self, x=None, y=None, z=None, num=0,
               max_length_x=None, max_length_y=None):
        if x is None:
            x = self.data_x[num]
        if y is None:
            y = self.data_y[num]
        if z is None:
            z = self.data_z[num]
        # check if everything is a numpy array
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        # check if the z matrix is ordered correctly
        if (z.shape[0] == x.size) and (z.shape[1] == y.size):
            print('x and y dimension are flipped in the z matrix')
            z = z.transpose()
        elif (z.shape[0] == y.size) and (z.shape[1] == x.size):
            pass
        else:
            print('cannot update 3d source due to shape error')
            return

        # save the full data
        self.data_x[num] = x
        self.data_y[num] = y
        self.data_z[num] = z

        if time.time() - self.lastrefresh[num] > self.max_refresh:
            # handle autoscaling before cutting
            self.autoscale()

            # do not plot more than one point per pixel, or more than given
            # as parameter
            if max_length_y is None:
                max_length_y = self.height
            else:
                max_length_y = np.min([self.height, max_length_y])
            if max_length_x is None:
                max_length_x = self.width
            else:
                max_length_x = np.min([self.width, max_length_x])

            # reduce the size for increased plotting speed
            y, z = reduce_y_2d(
                y, z, max_length_y,
                self.plot.y_range.start, self.plot.y_range.end)
            x, z = reduce_x_2d(
                x, z, max_length_x,
                self.plot.x_range.start, self.plot.x_range.end)

            # update plot
            self.sources[num].update(x, y, z)
            self.lastrefresh[num] = time.time()


def reduce_x_1d(x, y, goal_length_x, x_start=-np.inf, x_end=np.inf):
    return reduce_x_1d_c(
        np.array(x, dtype='float64'),
        np.array(y, dtype='float64'),
        goal_length_x, x_start, x_end)


@jit(nopython=True)
def reduce_x_1d_c(x, y, goal_length_x, x_start, x_end):
    y = y[(x >= x_start) & (x <= x_end)]
    x = x[(x >= x_start) & (x <= x_end)]

    old_x_size = x.size
    if (old_x_size > goal_length_x):
        factor = int(np.floor(old_x_size/goal_length_x))
        new_x_size = int(np.floor(x.shape[0]/factor))
        x_disp = np.zeros(new_x_size)
        y_disp = np.zeros(new_x_size)
        for cntx in range(new_x_size):
            x_disp[cntx] = x[cntx*factor:(cntx+1)*factor].sum()
            y_disp[cntx] = y[cntx*factor:(cntx+1)*factor].sum()
        x_disp = x_disp/factor
        y_disp = y_disp/factor
    else:
        x_disp = x
        y_disp = y
    return x_disp, y_disp


def reduce_y_2d(y, z, goal_length_y, y_start=-np.inf, y_end=np.inf):
    z = reduce_y_2d_c(
        np.array(y, dtype='float64'),
        np.array(z, dtype='float64'),
        goal_length_y, y_start, y_end)
    y = reduce_x_1d(
        x=y,
        y=y,
        goal_length_x=goal_length_y, x_start=y_start, x_end=y_end)
    return y, z


def reduce_x_2d(x, z, goal_length_x, x_start=-np.inf, x_end=np.inf):
    x, z = reduce_y_2d(
        y=x,
        z=z.transpose(),
        goal_length_y=goal_length_x,
        y_start=x_start, y_end=x_end)
    z = z.transpose()
    return x, z


@jit(nopython=True)
def reduce_y_2d_c(y, z, goal_length_y, y_start, y_end):
    z = z[(y >= y_start) & (y <= y_end), :]
    # y = y[(y >= y_start) & (y <= y_end)]

    old_y_size = z.shape[0]
    x_size = z.shape[1]
    if (old_y_size > goal_length_y):
        factor = int(np.floor(old_y_size/goal_length_y))
        new_y_size = int(np.floor(old_y_size/factor))
        z_disp = np.zeros((new_y_size, x_size))
        # y_disp = np.zeros(new_y_size)
        for cntx in range(x_size):
            for cnty in range(new_y_size):
                z_disp[cnty, cntx] = (
                    z[cnty*factor:(cnty+1)*factor, cntx].sum())
                # y_disp[cnty] = y[cnty*factor:(cnty+1)*factor].sum()
        # y_disp = y_disp/factor
        z_disp = z_disp/factor
    else:
        # y_disp = y
        z_disp = z
    return z_disp
