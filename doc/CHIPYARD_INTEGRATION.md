# Chipyard Integration

MiCo supports the designs from Chipyard, the following ones have been tested:
+ Rocket CPU
+ Boom CPU
+ Gemmini (INT8)
+ Spike Simulator

For other designs, it should be no problem, as long as the correct compiling flags are set for the `project`. For a working chipyard installation, just activate the chipyard conda environment, and build MiCo project as usual. Using `TARGET=rocket` or `TARGET=spike` should be enough for most cases, but for Gemmini you need to set `CHIPYARD_DIR=<your chipyard path>` as well.
