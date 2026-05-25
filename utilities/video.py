import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utilities.render_utils import get_pil_image_from_tensor


class Video:
    def __init__(self, path, name='video_log.mp4', mode='I', fps=30, codec='libx264', bitrate='16M') -> None:
        if path[-1] != "/":
            path += "/"

        self.writer = imageio.get_writer(path + name, mode=mode, fps=fps, codec=codec, bitrate=bitrate)

    def ready_image(self, image, loss, loss_dict, step_index, write_video=True, font_size=14):
        """
        Adds log information (loss, loss_dict, step index) to the image and appends it to the video.

        Args:
            image (torch.Tensor or np.ndarray): The image to process and append to the video.
            loss (float): The loss value to display.
            loss_dict (dict): The loss dictionary to display.
            step_index (int): The step index to display.
            write_video (bool): Flag to determine if the image should be appended to the video.
            font_size (int): The desired font size for the text.

        Returns:
            image (np.ndarray): The processed image with log information added.
        """
        # Convert tensor to numpy if the image is in torch tensor format
        image, pil_image = get_pil_image_from_tensor(image)
        draw = ImageDraw.Draw(pil_image)

        # Use a larger font size (you can replace with any .ttf file you like)
        try:
            font = ImageFont.truetype("Monaco.ttf", font_size)  # You can change to a specific font
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if the custom font is not found

        # Define text and background color
        text_color = (255, 255, 255)  # White text
        loss_info_color = (255, 255, 255)  # White for loss information
        background_color = (0, 0, 0)  # Black background for text (for visibility)

        # Text formatting (show loss, loss dict, step index)
        text = f"Step: {step_index} | Loss: {loss:.4f}" if loss is not None else ""
        loss_info = "".join([f"{key}: {val:.4f}\n" for key, val in loss_dict.items()]) if loss_dict else ""

        # Position the text (ensure it's within the image bounds)
        padding = 10
        text_position = (padding, padding)
        loss_info_position = (padding, padding + font_size + 10)

        # Calculate text width and height for background rectangle sizing
        text_bbox = draw.textbbox((0, 0), text, font=font)
        loss_info_bbox = draw.textbbox((0, 0), loss_info, font=font)

        # Add background behind text (a rectangle with padding around the text)
        background_margin = 5
        draw.rectangle([text_position, (text_position[0] + text_bbox[2] - text_bbox[0] + 2 * background_margin,
                                       text_position[1] + text_bbox[3] - text_bbox[1] + background_margin)],
                       fill=background_color)
        draw.rectangle([loss_info_position, (loss_info_position[0] + loss_info_bbox[2] - loss_info_bbox[0] + 2 * background_margin,
                                             loss_info_position[1] + loss_info_bbox[3] - loss_info_bbox[1] + background_margin)],
                       fill=background_color)

        # Add the text to the image (draw)
        draw.text(text_position, text, font=font, fill=text_color)
        draw.text(loss_info_position, loss_info, font=font, fill=loss_info_color)

        # Convert the PIL image back to numpy for appending
        image = np.array(pil_image)

        # Optionally append the image to the video
        if write_video:
            self.writer.append_data(image)

        return image

    def background_and_txt(self,
                           text: str,
                           image_size=(720, 720),
                           font_size=24,
                           background_color=(0, 0, 0),
                           text_color=(255, 255, 255),
                           write_video=True):
        """
        Creates a single-frame image of given size with a solid background and centered text,
        then optionally appends it to the video.

        Args:
            text (str): Text to render (can include newlines).
            image_size (tuple): (width, height) in pixels.
            font_size (int): Font size for rendering.
            background_color (tuple): RGB background color.
            text_color (tuple): RGB text color.
            write_video (bool): If True, appends this frame to the video.

        Returns:
            np.ndarray: The generated frame as an H×W×3 array.
        """
        W, H = image_size
        pil_image = Image.new('RGB', (W, H), color=background_color)
        draw = ImageDraw.Draw(pil_image)

        try:
            font = ImageFont.truetype("Monaco.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Compute text block size
        lines = text.split("\n")
        line_heights = []
        max_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            line_heights.append(h)
            max_width = max(max_width, w)
        total_height = sum(line_heights) + (len(lines) - 1) * 5

        # Start drawing centered
        y_offset = (H - total_height) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            x = (W - w) // 2
            draw.text((x, y_offset), line, font=font, fill=text_color)
            y_offset += line_heights[i] + 5

        frame = np.array(pil_image)
        if write_video:
            self.writer.append_data(frame)
            self.writer.append_data(frame)
            self.writer.append_data(frame)
        return frame

    def close(self):
        self.writer.close()
