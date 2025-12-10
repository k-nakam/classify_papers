# How to run

This is the example repository for the automated classification of the articles.

## How to set up
- You need to upload all the documents as PDF file. If you have appendix, please concatenate PDF files into one file.
- You also need the OpenAI API KEY. Please obtain it and upload it into .env file. The sample file is in sample.env (replace ########### with your API KEY)

## How to run
Run the followings:

    ```
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    python classification.py --input_dir "Input Directory NAME" --output_dir "Output Directory NAME"
    ```