import os
import pathlib
import json
import requests

def update_user_data(new_user_data, user_file):
    with open(user_file, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
    data.update(new_user_data)
    with open(user_file, 'w', encoding="utf-8" ) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parse_new_user_data(data, image_dir):
    user_id = str(data['user_id'])
    location = data.get("location", "")
    is_employee = data.get("is_employee", False)
    name = data.get("name")
    company = data.get("company", "")
    job_title = data.get("job_title", "")
    contact = data.get("contact")
    username = data.get("username")
    if 'images' in data:
        download_images(user_id, data['images'], image_dir)
    return {
        user_id: {
            "location": location,
            "is_employee": is_employee,
            "name": name,
            "company": company,
            "job_title": job_title,
            "contact": contact,
            "username": username
        }
    }

def download_images(user_id, image_urls, image_dir):
    image_paths = []
    pathlib.Path(os.path.join(image_dir, user_id)).mkdir(parents=True, exist_ok=True)
    for url in image_urls:
        filename = url.split('/')[-1]
        file_path = os.path.join(image_dir, user_id, filename)
        r = requests.get(url)
        open(file_path, 'wb').write(r.content)
        image_paths.append(file_path)
    print("Finish download images.")
    return image_paths

