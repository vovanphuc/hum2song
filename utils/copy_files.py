import os
import shutil
import logging

train_list = ['train/hum/0703.mp3',
'train/hum/2685.mp3',
'train/hum/2735.mp3',
'train/hum/2677.mp3',
'train/hum/2602.mp3',
'train/hum/2570.mp3',
'train/hum/1914.mp3',
'train/hum/1927.mp3',
'train/hum/1320.mp3',
'train/hum/2244.mp3',
'train/hum/1318.mp3',]
public_test_list = ['public_test/hum/0413.mp3']
private_test_list = []

def copy_fail_cleaning_data(src_path="raw_path", dst_path = "temp_dir", list_type="all"):
    if list_type == "all":
        fail_list = train_list + public_test_list
    elif list_type == "public_test":
        fail_list = public_test_list
    elif list_type == "private_test":
        fail_list = private_test_list
    else:
        fail_list = []
    
    for file in fail_list:
        if not os.path.isfile(os.path.join(src_path, file)):
            logging.warn(f"Not found {os.path.join(src_path, file)}")
            continue
        print(f"Copy {os.path.join(src_path, file)} to {os.path.join(dst_path, file)}")
        shutil.copyfile(os.path.join(src_path, file),
                        os.path.join(dst_path, file))

        if file.split("/")[0] == "train":
            song = os.path.join(file.split("/")[0], "song", file.split("/")[-1])
            shutil.copyfile(os.path.join(src_path, song),
                            os.path.join(dst_path, song))