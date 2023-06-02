import sys

from video import VideoFilter
from webcam import WebcamFilter

if __name__ == '__main__':
    args = sys.argv[1:]
    if (len(args) == 0):
        web_filter = WebcamFilter()
        web_filter.run()
    elif (len(args) == 1):
        vid_filter = VideoFilter()
        vid_filter.run(args[0])
    else:
        print('python ' + sys.argv[0] + ' [video_file]')
