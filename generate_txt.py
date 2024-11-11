import os

def file_name(file_dir):
    files = [os.path.splitext(file)[0].strip() for root, dirs, files in os.walk(file_dir) for file in files]
    with open(os.path.join(file_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(files))

    print('Generating Finished')

file_dir = '/home/ubuntu/usrs/JK/Pointfilter-master/Pointfilter-master/Dataset/GT/PUNet/meshes/test'
file_name(file_dir)