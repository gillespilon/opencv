#! /usr/bin/env python3

'''
OpenCV for beginners YouTube video

01 Introduction to OpenCV
03 Read, write, show images
04 Read, write, show videos
'''

import numpy as np
import cv2


def main():
    show_version()
    image_name_in, flag_value, window_name, image_name_out, video_name_out = \
        'data/lena.jpg', 1, 'lena', 'data/lena_copy.png', 'data/gilles.avi'
    image = read_image(file=image_name_in, flag_val=flag_value)
    print(type(image))
    show_image(windowname=window_name, image=image)
    write_image(file=image_name_out, image=image)
    capture_livestream_video(videoout=video_name_out, colour=True)


def show_version() -> None:
    '''
    Show version number of currently-installed OpenCV library.
    '''

    print(f'OpenCV version: {cv2.__version__}')


def read_image(
    file: str,
    flag_val: int
) -> None:
    '''
    Read an image.

    file    : the name of the image file
    flag_val: 0 for gray, 1 for colour
    '''

    img = cv2.imread(filename=file, flags=flag_val)
    return img


def show_image(
    windowname: str,
    image: np.ndarray
) -> None:
    '''
    Show image in a separate window.

    Press any key to close the resulting window.

    windowname: the name of the window in which to display the image
    file      : the numpy array of the image
    '''

    cv2.imshow(winname=windowname, mat=image)
    cv2.waitKey(delay=0)


def write_image(
    file: str,
    image: np.ndarray
) -> None:
    '''
    Write an image.

    file : the name of the file to write to
    image: the numpy array of the image
    '''

    cv2.imwrite(filename=file, img=image)


def capture_livestream_video(
    videoout: str,
    colour: bool = True
):
    '''
    Capture live video.

    videoout: the name of the video file to save
    colour  : True for colour, False for gray
    '''
    # VideoCapture.read() grabs, decodes, returns the bool, next video frame
    # Try -1, 0, 1 for your camera
    capture_frame = cv2.VideoCapture(index=0)
    fourcc_code = cv2.VideoWriter_fourcc(*'XVID')
    save_video = cv2.VideoWriter(
        filename=videoout,
        fourcc=fourcc_code,
        fps=10,
        frameSize=(1280, 720)
        )
    # capture_frame = cv2.VideoCapture(filename='videofilename.avi')
    while(True):
        # video_frame_bool is of type bool
        # True if a video frame is returned
        # False if a video frame is not returned
        # video_frame is of type np.ndarray
        video_frame_bool, video_frame = capture_frame.read()
        # To determine a property.
        if video_frame_bool is True:
            print(
                f'Frame width {capture_frame.get(cv2.CAP_PROP_FRAME_WIDTH)} '
                f'Frame height {capture_frame.get(cv2.CAP_PROP_FRAME_HEIGHT)} '
                f'fps {capture_frame.get(cv2.CAP_PROP_FPS)}'
            )
            save_video.write(video_frame)
            if colour is True:
                cv2.imshow(winname='colour_frame', mat=video_frame)
            elif colour is False:
                video_frame = cv2.cvtColor(
                    src=video_frame,
                    code=cv2.COLOR_BGR2GRAY
                )
                cv2.imshow(winname='gray_frame', mat=video_frame)
            # Press 'q' to stop video capture
            if cv2.waitKey(delay=1) == ord('q'):
                break
        else:
            break
    capture_frame.release()
    save_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
