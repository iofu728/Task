'''
@Author: gunjianpan
@Date:   2019-03-31 11:23:52
@Last Modified by:   gunjianpan
@Last Modified time: 2019-03-31 11:25:57
'''

import time

start = []


def begin_time():
    """
    multi-version time manage
    """
    global start
    start.append(time.time())
    return len(start) - 1


def end_time(version):
    timeSpend = time.time() - start[version]
    print('Cost: {:.2f}s.......'.format(timeSpend))
