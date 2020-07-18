import argparse

import file_utils
import ocr

def main(args):

    #make or clean the corresponding folders
    file_utils.make_clean_folder(args.intermediary_folder)
    file_utils.make_clean_folder(args.detection_folder)
    file_utils.make_clean_folder(args.recognition_folder)
    file_utils.make_clean_folder(args.image_folder)
    file_utils.make_clean_folder(args.ocr_result_folder)
    file_utils.make_clean_folder(args.output_folder)

    #convert all the pdfs into images and copy them along with other images into image folder
    file_utils.copy_and_convert(args.input_folder, args.image_folder)

    #run extraction on the entire folder
    ocr.extraction(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Extraction')

    """=================================== I/O Configuration ==================================="""

    parser.add_argument('--input_folder', default='input_folder/', type=str, help='folder path to input files')
    parser.add_argument('--intermediary_folder', default='intermediary_folder/', type=str, help='folder path to intermediary folders')
    parser.add_argument('--image_folder', default='intermediary_folder/images/', type=str, help='folder path to converted images')
    parser.add_argument('--detection_folder', default='intermediary_folder/detection/', type=str, help='folder path to detection results')
    parser.add_argument('--recognition_folder', default='intermediary_folder/recognition/', type=str, help='folder path to recognition results')
    parser.add_argument('--ocr_result_folder', default='intermediary_folder/ocr_result/', type=str, help='folder path to final ocr results (recognition+detection)')
    
    parser.add_argument('--output_folder', default='output_folder/', type=str, help='folder path to output files')
 

    """=================================== Part 1: DETECTION ==================================="""

    """ Data PreProcessing """
    parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')

    """ Data Loading Specifications """
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=3, help='input batch size')

    """ Detection Model Specifications """
    parser.add_argument('--trained_model', default='weights/detector_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold') #This threshold is not used in our case.
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score') #0.003 was used with 0.2-tt and 0.1-lt
    parser.add_argument('--link_threshold', default=0.1, type=float, help='link confidence threshold')
    parser.add_argument('--rotated_box', default=False, action='store_true', help='use this to get rotated rectangles (bounding box)') # Currently not handling for rotated boxes
    parser.add_argument('--x_dilate', default=1, type=int, help='left x-padding during post processing')
    parser.add_argument('--y_dilate', default=3, type=int, help='up y-padding during post processing')

    """ System Configuration """
    parser.add_argument('--cpu', default=False, action='store_true', help='Use cuda for inference')
    parser.add_argument('--verbose', default=False, action='store_true', help='show processing time and other details')
    parser.add_argument('--render', default=False, action='store_true', help='use to visualize intermediary results')

    """=================================== Part 2: RECOGNITION ==================================="""

    parser.add_argument('--recognize', default=False, action='store_true', help='flag to enable recognition')

    args = parser.parse_args()

    main(args)