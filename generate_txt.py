import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        with open(file_dir + '/test.txt', 'w') as f:
            for file in files:
                if file.endswith('.xyz'):
                    f.write(file.split('.')[0] + '\n')
file_dir = ''
file_name(file_dir)