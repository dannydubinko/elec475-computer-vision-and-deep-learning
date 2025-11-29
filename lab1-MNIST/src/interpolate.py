#########################################################################################################
#
#   ELEC 475 - Lab 1 - Interpolate Module
#   Fall 2025
#   Daniel Dubinko 
#   19dd34@queensu.ca

import torch

class Interpolate():
    def __init__(self, img_1, img_2, num_steps, model):
        """Interpolate from an initial image to a final image

        Args:
            img_1 (tensor): inital image
            img_2 (tensor): final image
            num_steps (int): desired number of interpolation steps
            model (autoencoderMLP4Layer): model used for encoding and decoding
        """
        self.img_1 = img_1
        self.img_2 = img_2
        self.num_steps = num_steps
        self.model = model

    def apply(self):
        encoded1 = self.model.encode(self.img_1)
        encoded2 = self.model.encode(self.img_2)
		
        images = []
        for i in range(self.num_steps):
            # interpolate through the 2 tensors for n steps, creating a set of n new bottleneck tensors
            alpha = i / (self.num_steps - 1)
            interpolate_image = torch.lerp(encoded1, encoded2, alpha)

            with torch.no_grad():
                # Pass each of these new tensors through the decode method
                decoded_image = self.model.decode(interpolate_image)

            # Plot the results
            decoded_image = decoded_image.reshape(28, 28)
            images.append(decoded_image)
        return images