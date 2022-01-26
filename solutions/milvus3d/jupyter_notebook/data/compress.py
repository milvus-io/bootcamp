import os
from collections import deque
import pymeshlab as ml
import getopt
import sys

sys.path.append("..")

from notebook_config import SEARCH_FEATURE_PATH, LOAD_FEATURE_PATH, UPLOAD_PATH


def process(data_root, out_root, file_name):
    """
    Preprocess and compress a single 3d models
    Args:
        data_root ([str]): path of where the 3d models are originally stored
        out_root ([str]): path of where the compressed 3d models will be stored
        file_name ([str]): file name of the current 3d model
    """
    # check whether the off file is valid
    # For some reason, there are some broken files in the original ModelNet40 dataset

    if file_name.endswith('.off'):
        try:
            in_path = os.path.join(data_root, file_name)
            out_path = os.path.join(out_root, file_name)

            with open(in_path) as f:
                first_line = f.readline()
                # check whether the off file is valid
                if len(first_line) != 3:
                    data = open(in_path, 'r').read().splitlines()
                    data = pre_process(data=data)
                    write_file(data=data,out_path=in_path)
        except Exception as e:
            print(" Error with check off file: {}".format(e))
            sys.exit(1)
    else:
        in_path = os.path.join(data_root, file_name)
        out_path = '.'.join(os.path.join(out_root, file_name).split('.')[:-1]) + '.off'
    try:
        # load
        ms = ml.MeshSet()
        ms.load_new_mesh(in_path)

        # turns a polygon mesh into a pure triangular mesh
        ms.turn_into_a_pure_triangular_mesh()

        # compress to 1024 faces
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=1024, preservenormal=True)
        ms.save_current_mesh(out_path, save_face_color=False)

    except Exception as e:
        print("Error with compressing off file: {}".format(e))
        sys.exit(1)


def write_file(out_path, data):
    with open(out_path, 'w') as f:
        for x in data:
            f.write(str(x) + '\n')


def pre_process(data):
    data = deque(data)
    first_element = data.popleft()
    firstline, secondline= first_element[0:3], first_element[3:]
    
    data.appendleft(secondline)
    data.appendleft(firstline)
    data = list(data)
    return data


def main(data_root):
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "batch=", "path=", "filename="])
    run_type, filename, path = [None] * 3

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("Help yourself please :)")
        elif opt_name == "--batch":
            run_type = opt_value
        elif opt_name == "--filename":
            filename = opt_value
        elif opt_name == "--path":
            path = opt_value

    if run_type == "T":
        run_batch(data_root, LOAD_FEATURE_PATH)
    elif run_type == "F":
        if filename:
            run_single(path, SEARCH_FEATURE_PATH, filename)
        else:
            print("Please provide filename to run a single file")
    else:
        print("T or F only")


def run_single(path, out_root, filename):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    process(data_root=path, out_root=out_root, file_name=filename)


def run_batch(data_root, out_root):
    print("start batch")
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    for file_name in os.listdir(data_root):
        print(file_name)
        if file_name.endswith("off"):
            process(data_root=data_root, out_root=out_root, file_name=file_name)


if __name__ == "__main__":
    print(os.getcwd())
    main(UPLOAD_PATH)
