import pickle
from tag import Tag

# Define the paths to the files
img2others_path = './Data/image_to_attributes_full.pkl'
images_filenames_path = './Data/images_filenames.pkl'
tags_path = './Data/tags.pkl'

if __name__ == '__main__':
    # Load the data from image_to_attributes_full.pkl
    with open(img2others_path, 'rb') as f:
        data = pickle.load(f)

    # Get the static tags from the Tag class
    tags = Tag().static_tags
    print('Number of tags:', len(tags))

    # Initialize a counter for the number of tags found in all the images
    num = 0

    # Load the filenames from images_filenames.pkl
    with open(images_filenames_path, 'rb') as f:
        filenames = pickle.load(f)

    # Initialize a dictionary to store the data to save
    data_to_save = {}

    # Iterate over the filenames
    for each in filenames:
        # Strip the filename and use it as the key
        key = each.strip()

        # Get the value associated with the key from the data
        value = data[key][-1]

        # Initialize a list for the key in data_to_save
        data_to_save[key] = []

        # If a tag is not in the static tags, add 'others' to the value
        for tag in value:
            if tag not in tags:
                value.append('others')

        # If a tag is in the value, increment the counter and append 1 to data_to_save[key]
        # Otherwise, append 0 to data_to_save[key]
        for tag in tags:
            if tag in value:
                num += 1
                data_to_save[key].append(1)
            else:
                data_to_save[key].append(0)

    # Save data_to_save to tags.pkl
    with open(tags_path, 'wb') as output:
        pickle.dump(data_to_save, output)