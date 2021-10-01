import os

if os.path.isfile('train.py') and os.path.isfile('datapreprocessing.py'):
    if os.path.exists("temp/"):
        print("temp folder already exists.")
    else:
        os.mkdir("temp/")
        print("temp folder created successfully.")

    if os.path.exists("data/"):
        print("data folder already exists.")
    else:
        os.mkdir("data/")
        print("data folder created successfully.")
    
    if os.path.exists("logs/"):
        print("logs folder already exists")
    else:
        os.mkdir("logs/")
        print("logs folder created successfully.")

else:
    raise IOError("Not under the main directory. Please go back to the main directory then excute this script.")