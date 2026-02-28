import matplotlib.pyplot as plt
import numpy as np

def plot_label_and_image(pred_mask_img, real_mask_img, combined_img: np.ndarray) -> None:

    n_slice = np.where(real_mask_img > 0)[2][0] 

    plt.subplot(231)
    plt.imshow(combined_img[0, n_slice,:,:], cmap='gray')
    plt.title('Image flair')
    plt.subplot(232)
    plt.imshow(combined_img[1, n_slice,:,:], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(233)
    plt.imshow(combined_img[2, n_slice,:,:], cmap='gray')
    plt.title('Image t2')
    plt.subplot(234)
    plt.imshow(pred_mask_img[n_slice,:,:])
    plt.title('Pred Mask')
    plt.subplot(235)
    plt.imshow(real_mask_img[n_slice,:,:])
    plt.title('Real Mask')
    plt.show()

def plot_graph(y_array : list, graph_name, x_label, y_label : str) -> None:

    x_array = [i for i in range(1, len(y_array) + 1)]
    plt.plot(x_array, y_array)
    plt.title(graph_name)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()