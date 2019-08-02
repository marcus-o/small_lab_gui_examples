import bokeh
import bokeh.layouts
import bokeh.plotting

import bokeh.server.server
import bokeh.application
import bokeh.application.handlers.function

import socket


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
