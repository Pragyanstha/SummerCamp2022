from email.policy import default
import configargparse

def get_config(args=None):
    p = configargparse.ArgParser(default_config_files=['configs/default.ini'])
    p.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    p.add('--data_dir', required=True, help='path to input files')  # this option can be set in a config file because it starts with '--'
    p.add('--result_dir', required=True, help='path to output files')  # this option can be set in a config file because it starts with '--'
    p.add('--algo', required=True, type=str, default="baseline", help="name of algorithm to run")
    p.add('--score_th', required=False, type=float, default=0.9, help="score_th")
    p.add('-v', help='verbose', action='store_true')
    p.add('--device', help='CUDA gpu number', type=str)
    p.add('--fps', type=int, default=5, help="FPS to read the video file")
    p.add('--weight_file', type=str, help="path to weight file")
    p.add('--config_file', type=str, help="path to config file for mmdetection models")
    if args is not None:
        return  p.parse_args(args)
    return p.parse_args()
