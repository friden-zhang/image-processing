import argparse
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_geometric_image(width, height):
    # Create an RGB image with a white background
    image = Image.new("RGB", (width, height), (255, 255, 255))

    # Draw a red rectangle within image bounds
    for x in range(50, min(250, width)):
        for y in range(50, min(150, height)):
            image.putpixel((x, y), (255, 0, 0))

    # Draw a blue circle within image bounds
    for x in range(300, min(500, width)):
        for y in range(50, min(250, height)):
            if (x - 400)**2 + (y - 150)**2 <= (100**2):
                image.putpixel((x, y), (0, 0, 255))

    # Draw a green line within image bounds
    for x in range(50, min(500, width)):
        if 200 < height:  # Ensure y-coordinate is within bounds
            image.putpixel((x, 200), (0, 255, 0))

    return image

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Create a geometric RGB image in raw format.")
    parser.add_argument("--width", type=int, help="Width of the image")
    parser.add_argument("--height", type=int, help="Height of the image")
    
    args = parser.parse_args()

    width = args.width
    height = args.height

    # Create the image
    image = create_geometric_image(width, height)

    # Save the image in raw RGB format
    with open("/tmp/geometric_image.rgb", "wb") as f:
        f.write(image.tobytes())
    logging.info("Image saved as raw RGB format: /tmp/geometric_image.rgb")

if __name__ == "__main__":
    main()
