import json
import zipfile
import os
import sys

def main():
    filename = sys.argv[1:][0] # 取得 python UnzipFile.py 後面的 args
    # print(filename)
    # print(type(filename))

    zip_ref = zipfile.ZipFile('./'+str(filename)+'.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()

if __name__ == "__main__":
    main()