import os
import time
import multiprocessing as mp
import torchvision.transforms as transforms
from PIL import Image

def process_image(filepath,savepath):
    img = Image.open(filepath).convert("RGB")
    width,height = img.size
    transform = transforms.Compose([
        transforms.CenterCrop(size=(height, height)),
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
    ])
    img = transform(img)
    transforms.functional.to_pil_image(img).save(savepath)

if __name__ == '__main__':
    test_dir = "new_test"
    new_test_dir = "new_test_resize512"
    os.makedirs(new_test_dir, exist_ok=True)

    cnt = 0
    start = time.time()
    tasks = mp.Queue()
    pool = mp.Pool(processes=6)

    for filename in os.listdir(test_dir):

        filepath = os.path.join(test_dir, filename)
        savepath = os.path.join(new_test_dir, filename)

        tasks.put((filepath,savepath))

    results = [pool.apply_async(process_image, t) for t in iter(tasks.get, 'STOP')]

    for r in results:
        r.get()

    end = time.time()
    print("Total time:", end-start)
