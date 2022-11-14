Please extract the zip file to get the ActDial dataset. The dialogue sessions of ActDial come from ConvAI2 and DailyDialog. Please refer to Section 3 of the original paper for detailed descriptions.

A list of dictionary can be obtained after reading each json file. The keys of all dictionaries are "index", "fs_text", "annotation", and "annotation_raw".
An index takes the form of "{dataset}_{split}_{dialogID}_{utteranceID}_{segmentID}". 
The value of "fs_text" is the raw text of the segment.
Three raw annotations are listed as the "annotation_raw" and the majority vote is the value of "annotation".

Example:
{
    "index": "persona_train_1_0_0",
    "fs_text": "hi ",
    "annotation": "greeting",
    "annotation_raw": [
        "greeting",
        "greeting",
        "greeting"
    ]
}

This entry comes from the training split of the ConvAI2 dataset. It is the 1st segment of the 1st utterance in the 1st dialogue session.
The raw content is "hi " and its segment act label is greeting.
All three annotators label it as greeting.