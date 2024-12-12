# bench


# 2D Tile Registers

The 2D tile registers are each 1024 bytes in size, and there are 8 of them. These registers 
have 16 rows each of 64 bytes. For BF16 this means, a matrix of 16 * 32 BF16 numbers 
whereas for INT8, this means a matrix of 16 * 64 INT8 numbers.
