import torch

class Reconstruction():

    @staticmethod
    def select_loss_pixel(loss_pixel='mse'):

        if(loss_pixel == 'mse'):
            return Reconstruction.Pixel.mse()
        elif(loss_pixel == 'l1'):
            return Reconstruction.Pixel.l1()

        return None

    class Pixel():
        
        # MSE loss
        @staticmethod
        def mse():
            return torch.nn.MSELoss()

        # L1 Loss
        @staticmethod
        def l1():
            return torch.nn.L1Loss()