import os

# data Path
data_dir = "../data/raw-data"

for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    print(category_path)
    if os.path.isdir(category_path):
        files = sorted(os.listdir(category_path))
        print(files)
        count = 1
        for file in files:
            file_path = os.path.join(category_path, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1]
                new_name = f"{category}{count}{ext}"
                new_path = os.path.join(category_path, new_name)
                print(new_path)
                # avoid duplicate
                while os.path.exists(new_path):
                    count += 1
                    new_name = f"{category}{count}{ext}"
                    new_path = os.path.join(category_path, new_name)

                os.rename(file_path, new_path)
                count += 1
print("Successfully rename all the images!")
