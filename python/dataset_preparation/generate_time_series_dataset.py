import os
import matplotlib.pyplot as plt
data_dir = 'data'
class_dirs = os.listdir(data_dir)

def get_track_info(file_name):
    if '(' in file_name:
        return None
    try:
        uuid, track_num, frame_num = file_name.split('_')
    except ValueError:
        return None
    return (
        uuid, 
        int(track_num, 10), 
        int(frame_num.split('.')[0], 10),
        file_name
        )
        

def sort_by_class():
    data = {}
    for class_dir in class_dirs:
        class_example_files = os.listdir(os.path.join(data_dir, class_dir))
        data[class_dir] = []  # Initialize an empty list for the class directory key
        for example_file_name in class_example_files:
            track_info = get_track_info(example_file_name)  # Store the tuple in a separate variable
            if track_info is None:
                continue
            data[class_dir].append(track_info)
            
    return data

def process_class(class_data):
    tracks = {}
    for class_example in class_data:
        if class_example[0] not in tracks:
            tracks[class_example[0]] = []
        tracks[class_example[0]].append(class_example)
    return tracks

data = sort_by_class()

for class_dir, class_data in data.items():
    print ("Number of pictures in class:", class_dir, ":", len(class_data))
    data[class_dir] = process_class(class_data)

#save the data to a folder hierarchy
destination_dir = 'data_time_series'

#copy files dont create simlinks
for class_dir, class_data in data.items():
    print ("Number of tracks in class:", class_dir, ":", len(class_data))
    for track_id, track_data in class_data.items():
        track_data.sort(key=lambda x: x[2])
        track_dir = os.path.join(destination_dir, class_dir, track_id)
        os.makedirs(track_dir, exist_ok=True)
        for frame in track_data:
            os.symlink(
                os.path.join('..', '..', '..', 'data', class_dir, frame[3]),
                os.path.join(track_dir, frame[3])
                )

