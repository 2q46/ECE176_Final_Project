from dataclasses import dataclass

@dataclass
class UNetParams:

    # Encoder params
    encoder_kernel : tuple = (3, 3, 3)
    encoder_stride : int = 1
    encoder_padding : int = 1

    # Decoder params
    decoder_kernel : tuple = (2, 2, 2)
    decoder_stride : int = 2

    # misc
    max_pool_kernel : tuple = (2, 2, 2)
    final_kernel : tuple = (1, 1, 1)
    final_conv_in : int = 16
    final_conv_out : int = 4

    # bottleneck
    bottleneck_in : int = 128
    bottleneck_out : int = 256

    # Encoder
    encoder1_in : int = 3
    encoder1_out : int = 16

    encoder2_in : int = 16
    encoder2_out : int = 32

    encoder3_in : int = 32
    encoder3_out : int = 64

    encoder4_in : int = 64
    encoder4_out : int = 128

    decoder1_in : int = 256   
    decoder1_out : int = 128

    decoder2_in : int = 128  
    decoder2_out : int = 64

    decoder3_in : int = 64 
    decoder3_out : int = 32

    decoder4_in : int = 32  
    decoder4_out : int = 16