import s3_access

if __name__ == "__main__":
    fs = s3_access.fs
    print(fs.ls("simulations"))