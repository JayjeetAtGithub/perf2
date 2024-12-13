# bench

## 2D Tile Registers

The 8 2D tile registers are each 1KiB in size. These registers 
have 16 rows each of 64 bytes. For BF16 this means, a matrix of 16 * 32 BF16 numbers 
whereas for INT8, this means a matrix of 16 * 64 INT8 numbers.

## System Configuration

* Sockets: 2
* CPU: Intel Xeon Gold 5418Y CPU / 24 cores per socket / 48 cores total / SMT-enabled
* Memory: 256 GB / DDR5 / 4800 MT/s

## AMX Configuration

1. TMUL: 48 (1 per core)
2. Tiles: 48 * 8 * 1KiB = 384 KiB (1 per TMUL)
