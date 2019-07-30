class elog_dummy:
    """post to elog dummy
    """
    def __init__(self, host='localhost', logbook='demo', port=8080):
        pass

    def post(self, author, subject, text, filename=None):
        print('dummy post')
