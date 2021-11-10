import numpy as np
import os
import pymesh
import traceback
import sys
import getopt
sys.path.append("..")

from src.config import SEARCH_FEATURE_PATH, LOAD_FEATURE_PATH, DATA_PATH


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            try:
                face.remove(vf1)
                face.remove(vf2)
            except:
                continue
            return i

    return except_face


def extract_features(in_path, file_name):
    """
    Extract feature for a single compressed 3d model
    Args:
        data_root ([str]): path of where the preprocessed 3d models are stored
        out_root ([str]): path of where the extracted features (npy file) will be stored
        file_name ([str]): file name of the current 3d model
    """
    
    try:
        path = os.path.join(in_path, file_name)
        mesh = pymesh.load_mesh(path)

        # delete the file after load
        os.remove(path)

        # clean up
        mesh, _ = pymesh.remove_isolated_vertices(mesh)
        mesh, _ = pymesh.remove_duplicated_vertices(mesh)

        # get elements
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()

        # move to center
        center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
        vertices -= center

        # normalize
        max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
        vertices /= np.sqrt(max_len)

        # get normal vector
        mesh = pymesh.form_mesh(vertices, faces)
        mesh.add_attribute('face_normal')
        face_normal = mesh.get_face_attribute('face_normal')

        # get neighbors
        faces_contain_this_vertex = []
        for i in range(len(vertices)):
            faces_contain_this_vertex.append(set([]))
        centers = []
        corners = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            x1, y1, z1 = vertices[v1]
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]
            centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
            corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        neighbors = []
        for i in range(len(faces)):
            [v1, v2, v3] = faces[i]
            n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])

        centers = np.array(centers)
        corners = np.array(corners)
        faces = np.concatenate([centers, corners, face_normal], axis=1)
        neighbors = np.array(neighbors)
        _, filename = os.path.split(file_name)
        filename = in_path + '/' + filename[:-4] + '.npz'

        # save as npy
        np.savez(filename, faces=faces, neighbors=neighbors)
        print(f"saved {filename}")

        # return the path of the saved npy file
        return filename
        
    except Exception as exc:
        print(f"{file_name} broken")
        traceback.print_exc()


def run_batch(in_path):

    for file_name in os.listdir(in_path):
        if file_name.endswith("off"):
            extract_features(in_path, file_name)
    print("success!")


def run_single(in_path, filename):
    extract_features(in_path, filename)


def main(search_path, load_path):
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "batch=", "filename="])
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("Help yourself please :)")
        elif opt_name == "--batch":
            run_type = opt_value
        elif opt_name == "--filename":
            filename = opt_value
    if run_type == "T":
        run_batch(load_path)
    elif run_type == "F":
        if filename:
            run_single(search_path, filename)
        else:
            print("where is the file name?")
    else:
        print("T or F")


if __name__ == '__main__':
    main(SEARCH_FEATURE_PATH, LOAD_FEATURE_PATH)
