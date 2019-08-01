from threading import Thread, Event
from functools import partial
import zmq
import json
import time


class asyncZmqHandler(Thread):
    def __init__(self, messages=[], listento='*', port=5557, bokeh_doc=None):
        # messages should be a list of string/callback tuples
        super().__init__()
        self.messages = messages
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind('tcp://' + listento + ':' + str(port))
        self.port = port
        self.bokeh_doc = bokeh_doc
        # self.stop_event = Event()

    def addMessage(self, message):
        # dont use this while server is running
        self.messages.append(message)

    def stop(self):
        # self.stop_event.set()
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect('tcp://localhost:' + str(self.port))
        self.socket.send_string('quitZmqHandler')

    def close(self):
        print('closing zmq handler')
        self.stop()

    def run(self):
        while True:
            message = self.socket.recv_string()
            if message == 'quitZmqHandler':
                break
            dec = json.loads(message)
            command = dec['command']
            value = dec['value']
            success = True
            for msg in self.messages:
                if msg[0] == command:
                    if value == '':
                        call = msg[1]
                    else:
                        call = partial(msg[1], value)
                    if self.bokeh_doc:
                        ret = {'ret': None}
                        stop_event = Event()

                        async def w(call, ret, stop_event):
                            ret['ret'] = call()
                            stop_event.set()

                        self.bokeh_doc.add_next_tick_callback(
                            partial(
                                w, call=call, ret=ret, stop_event=stop_event))
                        while not stop_event.is_set():
                            time.sleep(0.01)
                        ret = ret['ret']
                    else:
                        ret = call()
                    break
            else:
                ret = 0
                success = False
            enc = json.dumps({'command': command,
                              'value': ret,
                              'success': success})
            self.socket.send_string(enc)
        print('zmq thread ended')
        return(0)


class zmqRequester():
    def __init__(self, address='localhost', port=5557):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect('tcp://' + address+':' + str(port))

    def send(self, command, value=''):
        enc = json.dumps({'command': command, 'value': value})
        self.socket.send_string(enc)
        message = self.socket.recv_string()
        dec = json.loads(message)
        successRet = dec['success']
        commandRet = dec['command']
        valueRet = dec['value']
        if commandRet == command and successRet:
            return valueRet
        else:
            return False

    def close(self):
        print('closing zmq requester')
