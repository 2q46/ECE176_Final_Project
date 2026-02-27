
import matplotlib.pyplot as plt
import numpy as np

def plot_label_and_image(mask_img, combined_img: np.ndarray) -> None:

    n_slice = np.where(mask_img > 0)[2][0] 
    n_slice = 56

    plt.subplot(221)
    plt.imshow(combined_img[0, n_slice,:,:], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(combined_img[1, n_slice,:,:], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(combined_img[2, n_slice,:,:], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(mask_img[n_slice,:,:])
    plt.title('Mask')
    plt.show()

def plot_graph(y_array : np.ndarray, graph_name, x_label, y_label : str) -> None:

    x_array = [i for i in range(1, len(y_array) + 1)]
    plt.plot(x_array, y_array)
    plt.title(graph_name)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()