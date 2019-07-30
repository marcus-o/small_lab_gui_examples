from modules.digitizers.digitizer import digitizer

import serial


class pressure_controller_pfeiffer_tpg366(digitizer):
    def __init__(self, port, baud):
        self.num_sensors = 6
        try:
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=1)
            print('opened serial connection to ')
            print(self.ser)
        except Exception:
            print('could not open serial connection to pfeiffer maxigauge')

    def frame(self, stop_event=None, inp=None, init_output=None):
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.write("PRX\r\n".encode())
            self.ser.flush()
            if not self.ser.readline() == b'\x06\r\n':
                raise Exception('Unexpected serial answer')
            self.ser.write(b'\x05')
            self.ser.flush()
            p = self.ser.readline()
            stats = [0]*8
            prss = [1e8]*8
            valid = [True]*8
            if len(p) == 85:
                stats[0] = int(p[0])
                prss[0] = float(p[3:13])
                stats[1] = int(p[14])
                prss[1] = float(p[17:27])
                stats[2] = int(p[28])
                prss[2] = float(p[31:41])
                stats[3] = int(p[42])
                prss[3] = float(p[45:55])
                stats[4] = int(p[56])
                prss[4] = float(p[59:69])
                stats[5] = int(p[70])
                prss[5] = float(p[73:83])
                for idx, stat in enumerate(stats):
                    if not (stat == 48 or stat == 49):
                        prss[idx] = 1.e8
                        valid[idx] = False
                return {
                    'pressures': prss,
                    'valid': valid,
                    'source': 'tpg366',
                    'success': True}
        except Exception:
            return {
                'pressures': 0,
                'valid': 0,
                'source': 'tpg366',
                'success': False}

    def close(self):
        print('closing pfeiffer maxigauge')
        self.ser.close()
