from dataclasses import dataclass

@dataclass
class UNetParams:

    up_kernel_size: tuple = (3, 3, 3)
    down_kernel_size: tuple = (2, 2, 2)
    max_pool_kernel: tuple = (2, 2, 2)
    max_pool_stride: int = 2
    conv_stride: int = 1

    # encoder layer 1 
    encoder1_in: int = 3
    encoder1_out: int = 16
    # encoder layer 2
    encoder2_in: int = 32
    encoder2_out: int = 32
    # encoder layer 3
    encoder3_in: int = 64
    encoder3_out: int = 64
    # encoder layer 4
    encoder4_in: int = 128
    encoder4_out: int = 128

    # bottleneck
    bottleneck_in: int = 256
    bottleneck_out: int = 256

    # decoder layer 1  
    decoder1_in: int = 512
    decoder1_out: int = 128

    # decoder layer 2  
    decoder2_in: int = 128
    decoder2_out: int = 64

    # decoder layer 3  
    decoder3_in: int = 64
    decoder3_out: int = 32

    # decoder layer 4  
    deocder4_in: int = 32
    decoder4_out: int = 16

    # final conv
    final_conv_in: int = 16
    final_conv_out: int = 4 