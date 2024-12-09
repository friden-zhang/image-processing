import argparse
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_geometric_yuyv(width, height):
    # Create YUYV image with white background
    yuyv = np.full((height, width * 2), 255, dtype=np.uint8)

    # Draw a red rectangle within image bounds
    for x in range(50, min(250, width), 2):
        for y in range(50, min(150, height)):
            yuyv[y, x * 2] = 76  # Y1
            yuyv[y, x * 2 + 1] = 85  # U
            yuyv[y, x * 2 + 2] = 76  # Y2
            yuyv[y, x * 2 + 3] = 255  # V

    # Draw a blue circle within image bounds
    for x in range(300, min(500, width), 2):
        for y in range(50, min(250, height)):
            if (x - 400) ** 2 + (y - 150) ** 2 <= (100 ** 2):
                yuyv[y, x * 2] = 29  # Y1
                yuyv[y, x * 2 + 1] = 255  # U
                yuyv[y, x * 2 + 2] = 29  # Y2
                yuyv[y, x * 2 + 3] = 107  # V

    # Draw a green line within image bounds
    for x in range(50, min(500, width), 2):
        if 200 < height:
            yuyv[200, x * 2] = 150  # Y1
            yuyv[200, x * 2 + 1] = 44  # U
            yuyv[200, x * 2 + 2] = 150  # Y2
            yuyv[200, x * 2 + 3] = 21  # V

    return yuyv.tobytes()

def create_geometric_uyvy(width, height):
    # Create UYVY image with white background
    uyvy = np.full((height, width * 2), 255, dtype=np.uint8)

    # Draw a red rectangle within image bounds
    for x in range(50, min(250, width), 2):
        for y in range(50, min(150, height)):
            uyvy[y, x * 2] = 85  # U
            uyvy[y, x * 2 + 1] = 76  # Y1
            uyvy[y, x * 2 + 2] = 255  # V
            uyvy[y, x * 2 + 3] = 76  # Y2

    # Draw a blue circle within image bounds
    for x in range(300, min(500, width), 2):
        for y in range(50, min(250, height)):
            if (x - 400) ** 2 + (y - 150) ** 2 <= (100 ** 2):
                uyvy[y, x * 2] = 255  # U
                uyvy[y, x * 2 + 1] = 29  # Y1
                uyvy[y, x * 2 + 2] = 107  # V
                uyvy[y, x * 2 + 3] = 29  # Y2

    # Draw a green line within image bounds
    for x in range(50, min(500, width), 2):
        if 200 < height:
            uyvy[200, x * 2] = 44  # U
            uyvy[200, x * 2 + 1] = 150  # Y1
            uyvy[200, x * 2 + 2] = 21  # V
            uyvy[200, x * 2 + 3] = 150  # Y2

    return uyvy.tobytes()

def create_geometric_yvyu(width, height):
    # Create YVYU image with white background
    yvyu = np.full((height, width * 2), 255, dtype=np.uint8)

    # Draw a red rectangle within image bounds
    for x in range(50, min(250, width), 2):
        for y in range(50, min(150, height)):
            yvyu[y, x * 2] = 76  # Y1
            yvyu[y, x * 2 + 1] = 255  # V
            yvyu[y, x * 2 + 2] = 76  # Y2
            yvyu[y, x * 2 + 3] = 85  # U

    # Draw a blue circle within image bounds
    for x in range(300, min(500, width), 2):
        for y in range(50, min(250, height)):
            if (x - 400) ** 2 + (y - 150) ** 2 <= (100 ** 2):
                yvyu[y, x * 2] = 29  # Y1
                yvyu[y, x * 2 + 1] = 107  # V
                yvyu[y, x * 2 + 2] = 29  # Y2
                yvyu[y, x * 2 + 3] = 255  # U

    # Draw a green line within image bounds
    for x in range(50, min(500, width), 2):
        if 200 < height:
            yvyu[200, x * 2] = 150  # Y1
            yvyu[200, x * 2 + 1] = 21  # V
            yvyu[200, x * 2 + 2] = 150  # Y2
            yvyu[200, x * 2 + 3] = 44  # U

    return yvyu.tobytes()

def create_geometric_i420(width, height):
    # Create I420 image with white background
    y = np.full((height, width), 255, dtype=np.uint8)
    u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    v = np.full((height // 2, width // 2), 128, dtype=np.uint8)

    # Draw a red rectangle within image bounds
    for x in range(50, min(250, width)):
        for y_coord in range(50, min(150, height)):
            y[y_coord, x] = 76
            if x % 2 == 0 and y_coord % 2 == 0:
                u[y_coord // 2, x // 2] = 85
                v[y_coord // 2, x // 2] = 255

    # Draw a blue circle within image bounds
    for x in range(300, min(500, width)):
        for y_coord in range(50, min(250, height)):
            if (x - 400) ** 2 + (y_coord - 150) ** 2 <= (100 ** 2):
                y[y_coord, x] = 29
                if x % 2 == 0 and y_coord % 2 == 0:
                    u[y_coord // 2, x // 2] = 255
                    v[y_coord // 2, x // 2] = 107

    # Draw a green line within image bounds
    for x in range(50, min(500, width)):
        if 200 < height:
            y[200, x] = 150
            if x % 2 == 0:
                u[100, x // 2] = 44
                v[100, x // 2] = 21

    return y.tobytes() + u.tobytes() + v.tobytes()

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Create a geometric image in various formats.")
    parser.add_argument("--width", type=int, help="Width of the image")
    parser.add_argument("--height", type=int, help="Height of the image")
    parser.add_argument("--format", choices=["yuyv", "uyvy", "yvyu", "i420"], help="Output format")

    args = parser.parse_args()

    width = args.width
    height = args.height
    format = args.format

    # Create and save the image in the specified format
    if format == "yuyv":
        with open("/tmp/geometric_image.yuyv", "wb") as f:
            f.write(create_geometric_yuyv(width, height))
        logging.info("Image saved as YUYV format: /tmp/geometric_image.yuyv")
    elif format == "uyvy":
        with open("/tmp/geometric_image.uyvy", "wb") as f:
            f.write(create_geometric_uyvy(width, height))
        logging.info("Image saved as UYVY format: /tmp/geometric_image.uyvy")
    elif format == "yvyu":
        with open("/tmp/geometric_image.yvyu", "wb") as f:
            f.write(create_geometric_yvyu(width, height))
        logging.info("Image saved as YVYU format: /tmp/geometric_image.yvyu")
    elif format == "i420":
        with open("/tmp/geometric_image.i420", "wb") as f:
            f.write(create_geometric_i420(width, height))
        logging.info("Image saved as I420 format: /tmp/geometric_image.i420")

if __name__ == "__main__":
    main()