import numpy as np
import matplotlib.pyplot as plt


# Adapted from http://dohmatob.github.io/research/2016/10/22/VAE.html
def show_examples(data, mask, n=None, n_cols=20, mask_missing=True, img_rows=28, img_cols=20, figsize=(12, 10)):
    if n is None:
        n = len(data)
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    # Select samples
    for k, (x, m) in enumerate(zip(data[:n], mask[:n])):
        r = k // n_cols
        c = k % n_cols
        # Set image
        img_ref = figure[r * img_rows: (r + 1) * img_rows,
                         c * img_cols: (c + 1) * img_cols]

        image = x.reshape(img_rows, img_cols)
        if mask_missing:
            image = image * m.reshape(img_rows, img_cols)

        # Update image in figure
        img_ref[:, :] = image

    plt.figure(figsize=figsize)
    plt.imshow(figure, cmap='gray', interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()


def show_example(x, m, mask_missing=True):
    img_rows, img_cols = 28, 20

    image = x.reshape(img_rows, img_cols)
    if mask_missing:
        image = image * m.reshape(img_rows, img_cols)

    plt.figure(figsize=(4, 3.3))
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
