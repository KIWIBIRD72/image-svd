from PIL import Image, ImageFile
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from rich import inspect
import matplotlib.pyplot as plt


CURR_DIR = Path(__file__).parent
IMAGE_PATH = CURR_DIR / '..' / 'media' / 'smile.jpg'
IMAGE_PATH_2 = CURR_DIR / '..' / 'media' / 'mercury.jpeg'
K: int = 10  # Количество сингулярных значений для восстановления


def img_to_svd_decompose(image: ImageFile.ImageFile):
    grayscale_image = image.convert('L')

    vector_image = np.asarray(grayscale_image)

    TONES_RANGE = 255.0
    norm_range_vec_img = vector_image / TONES_RANGE

    # SVD разложение
    svd_res = np.linalg.svd(norm_range_vec_img, full_matrices=False)

    return (svd_res, norm_range_vec_img)


def reconstruct_img_from_svd(U: NDArray[np.float32], S: NDArray[np.float32], VT: NDArray[np.float32]) -> NDArray[np.float32]:
    rec_img = np.dot(U[:, :K], np.dot(np.diag(S[:K]), VT[:K, :]))
    return rec_img


def show_img(vec_img: NDArray[np.float32], orig_img: NDArray[np.float32]):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(orig_img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title(f'Reconstructed Image (k={K})')
    plt.imshow(vec_img, cmap='gray')
    plt.show()


def main():
    image = Image.open(IMAGE_PATH_2)
    svd_img, norm_vec_img = img_to_svd_decompose(image)
    rec_img = reconstruct_img_from_svd(svd_img.U, svd_img.S, svd_img.Vh)
    show_img(rec_img, norm_vec_img)


if __name__ == '__main__':
    main()
