import requests


class elog:
    """post to psi / stefan ritt elog labbooks (https://elog.psi.ch/elog/).
    usage
    a=elog(host='localhost', port=8080, logbook='demo')
    a.post(
        author='marcus', subject='neu', text='hello',
        filename='small_lab_gui/testfile.png')
    """
    def __init__(self, host='localhost', logbook='demo', port=8080):
        self.url = 'http://'+host+':'+str(port)+'/'+logbook

    def post(self, author, subject, text, filename=None):
        if filename is None:
            files = {'foo': ('bar', '')}
        else:
            files = {'attfile1': (filename, open(filename, 'rb'))}
        r = requests.post(
            "http://localhost:8080/demo",
            files=files,
            data={
                'Author': author,
                'Subject': subject,
                'Text': text,
                'encoding': 'plain',
                'next_attachment': '1',
                'cmd': 'Submit'
            })
        return r.status_code == 200
