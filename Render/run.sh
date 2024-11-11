#!/bin/bash

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

# Activate Conda virtual environment
conda activate mitsuba

# Specify the directory containing the XML files
directory="/media/ubuntu/JK的1号仓库/wcc实验/点云对照/mse"

# Iterate over all .xml files in the specified directory
for file in "$directory"/*.xml; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # Render the XML file using mitsuba
        mitsuba "$file"
    fi
done

# Deactivate the Conda virtual environment
conda deactivate