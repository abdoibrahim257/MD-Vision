import json
import pickle
import re
from tag import Tag

tags = Tag().static_tags

if __name__ == '__main__':
    
    # Load the data from the pickle file
    with open('./Data/image_to_attributes_full.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"Tags length: {len(tags)}")

    captions = {}

    # Process the captions
    for key in data:
        findings = data[key][0].replace('  ', ' ') # Replace double spaces with a single space
        discussion = data[key][1].replace('  ', ' ') # Replace double spaces with a single space
        caption = '. '.join([findings, discussion]) # Combine the findings and discussion
        caption = caption.replace(' .', '.').replace(',', '').replace('/', '.') # Remove spaces before periods and replace commas and slashes with periods

        # Use regex to replace all digits with '<num>'
        caption = re.sub(r'\d+', '<num>', caption)

        # Use regex to replace multiple occurrences of '<num>' with a single '<num>'
        caption = re.sub(r'(<num>)+', '<num>', caption)

        # Captions with less than 2 characters are replaced with 'normal'
        if len(caption) <= 2:
            caption = 'normal'
        
        # Add the caption to the dictionary
        captions[key] = [caption.strip() for caption in caption.split('.') if caption.strip()]

    # Save keys in a pkl file called images_filenames.pkl and .png is added to the key
    with open('./Data/images_filenames.pkl', 'wb') as f:
        pickle.dump(list(captions.keys()), f)

    # Save the captions as a JSON file
    with open('./Data/captions.json', 'w') as f:
        json.dump(captions, f , indent=4)

    # Save the captions as a pickle file
    with open('./Data/captions.pkl', 'wb') as f:
        pickle.dump(captions, f)

    print(f"Caption length: {len(captions)}")
    print('Captions saved')
    