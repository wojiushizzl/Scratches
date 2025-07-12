# 一个交互式对已有mask进行编辑的脚本
# # 显示mask和原图的叠加效果
# # 允许用户通过鼠标点击来添加或删除mask中的区域
# # button用于切换图片
# # button用于保存或撤销编辑
# # button用于删除image 和 对应的mask
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from glob import glob
from PIL import Image

class MaskEditor:
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))  # add bmp jpg png etc. if needed
        self.current_image = None
        self.current_index = 0

        if not self.image_paths:
            raise ValueError("No images found in the specified folder.")

        self.load_current_image()
        self.load_current_mask()

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.mask_display = None

        # Create buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.5, 0.05, 0.1, 0.075])
        ax_delete = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_undo = plt.axes([0.9, 0.05, 0.1, 0.075])
        ax_clean = plt.axes([0.1, 0.15, 0.1, 0.075])

        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_delete = Button(ax_delete, 'Delete')
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_clean = Button(ax_clean, 'Clean')

        # Connect the mouse click event to the erase function，允许长安拖动擦除
        self.cid = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.erase_mask)
        self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self.erase_mask)
        self.cid = self.fig.canvas.mpl_connect('button_release_event', self.release_mask)

        # Connect buttons to their functions
        self.btn_prev.on_clicked(self.show_previous_image)
        self.btn_next.on_clicked(self.show_next_image)
        self.btn_save.on_clicked(self.save_mask)
        self.btn_delete.on_clicked(self.delete_image_and_mask)
        self.btn_undo.on_clicked(self.undo_last_action)
        self.btn_clean.on_clicked(self.clean_mask)

        # Display the first image and mask
        self.update_display()

    def load_current_image(self):
        self.current_image_path = self.image_paths[self.current_index]
        print(f"Loading image from: {self.current_image_path}")
        self.current_image = cv2.imread(self.current_image_path)

    def load_current_mask(self):
        # mask_Image_20250510161059459.bmp
        mask_name = "mask_" + os.path.basename(self.current_image_path)
        mask_path = os.path.join(self.mask_folder, mask_name)
        print(f"Loading mask from: {mask_path}")
        if os.path.exists(mask_path):
            self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if self.current_mask is None:
                raise ValueError(f"Mask for {self.current_image_path} is not a valid image.")
            if len(self.current_mask.shape) == 3:
                # Convert to single channel if it's a color image
                print("Converting mask to grayscale.")
                self.current_mask = cv2.cvtColor(self.current_mask, cv2.COLOR_BGR2GRAY)
        else:
            self.current_mask = np.zeros_like(self.current_image[:, :, 0], dtype=np.uint8)
    def update_display(self):
        if self.mask_display is not None:
            self.mask_display.remove()

        # Create an overlay of the mask on the image
        if self.current_mask is None or self.current_image is None:
            raise ValueError("Current image or mask is not loaded properly.")
        # Create a color overlay for the mask  mask 在原图上显示为红色

        # 原图和mask图合并为一张图，利用mask高亮划痕区域
        img = self.current_image.copy()
        mask = self.current_mask.copy()
        # 将mask转换为3通道图像
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # 将mask_3channel中划痕区域设置为红色
        mask_3channel[mask == 255] = [0, 0, 255]
        # 将mask_3channel和img合并
        merged_img = cv2.addWeighted(img, 0.5, mask_3channel, 0.5, 0)
        # Display the image with the mask overlay

        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
        # self.ax.axis('off')
        # self.mask_display = self.ax.imshow(self.current_mask, alpha=0.5, cmap='jet')

        plt.draw()
    def show_previous_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            self.load_current_mask()
            self.update_display()
    def show_next_image(self, event):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_image()
            self.load_current_mask()
            self.update_display()
    def save_mask(self, event):
        mask_name = "mask_" + os.path.basename(self.current_image_path)
        mask_path = os.path.join(self.mask_folder, mask_name)
        cv2.imwrite(mask_path, self.current_mask)
        print(f"Mask saved to {mask_path}")

    def erase_mask(self, event):
        # 鼠标点击事件用于擦除鼠标位置直径为30范围内的mask
        # 允许长安拖动擦除，直到鼠标放开
        if event.inaxes != self.ax:
            return
        # Get the coordinates of the click
        x, y = int(event.xdata), int(event.ydata)
        # Check if the click is within the image bounds
        if (0 <= x < self.current_mask.shape[1]) and (0 <= y < self.current_mask.shape[0]):
            # Erase the mask in a 30-pixel radius around the click
            cv2.circle(self.current_mask, (x, y), 30, (0, 0, 0), -1)
            self.update_display()
            print(f"Erased mask at ({x}, {y})")
        else:
            print("Click is outside the image bounds.")
        # If the mouse is pressed, we can continue to erase
    def release_mask(self, event):
        # This function can be used to finalize the mask editing when the mouse is released.
        # For now, we will just print a message.
        print("Mouse released, mask editing finalized.")
        # You can implement additional logic here if needed.

    def clean_mask(self, event):
        # Clean the mask by removing all non-zero pixels
        if self.current_mask is not None:
            self.current_mask.fill(0)
            self.update_display()

    def delete_image_and_mask(self, event):
        image_path = self.image_paths[self.current_index]
        mask_path = os.path.join(self.mask_folder, "mask_" + os.path.basename(image_path))
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        if os.path.exists(mask_path):
            os.remove(mask_path)
            print(f"Deleted mask: {mask_path}")
        # Remove the current image from the list
        self.image_paths.pop(self.current_index)
        # If we deleted the last image, go back to the previous one
        if self.current_index >= len(self.image_paths):
            self.current_index = max(0, len(self.image_paths) - 1)
        # Reload the current image and mask
        self.load_current_image()
        self.load_current_mask()
        self.update_display()
    def undo_last_action(self, event):
        # This function can be implemented to undo the last action, such as restoring the previous mask state.
        # For simplicity, we will just print a message here.
        print("Undo last action is not implemented yet.")
        # You can implement a stack to keep track of changes and restore the previous state.
    def run(self):
        plt.show()
if __name__ == "__main__":
    image_folder = "./dataset/videos/images1"  # Adjust the path to your images folder
    mask_folder = "./dataset/videos/masks1"     # Adjust the path to your masks folder

    editor = MaskEditor(image_folder, mask_folder)
    editor.run()
# This script allows you to interactively edit masks for images in a specified folder.
# You can navigate through images, save edited masks, and delete images and their corresponding masks.