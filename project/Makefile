# Make

CC = gcc # gcc, clang
OBJDUMP = objdump

MICO_DIR = MiCo-Lib

MICO_INCLUDES = $(MICO_DIR)/include $(MICO_DIR)/test
MICO_SOURCES = $(wildcard $(MICO_DIR)/src/*.c)
MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico/*.c)

OPT ?=
TARGET ?= host # host, vexii

ifneq ($(filter unroll, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_unrolled/*.c)
else ifneq ($(filter im2col, $(OPT)),)
	MICO_SOURCES += $(wildcard $(MICO_DIR)/src/im2col_conv2d/*.c)
endif

CFLAGS = -O3
LDFLAGS = -lm

INCLUDES = $(MICO_INCLUDES) ./

BUILD = build
MAIN = main

# RISC-V
MABI?=ilp32
MARCH?=rv32imc
RISCV_PREFIX = riscv64-unknown-elf
RISCV_SOURCE ?= 

# VexiiRiscv Related
VEXII_PATH = $(MICO_DIR)/targets/vexii
VEXII_LD = $(MICO_DIR)/targets/vexii/vexii.ld

ifeq ($(TARGET), vexii)
	CC = $(RISCV_PREFIX)-gcc
	OBJDUMP = $(RISCV_PREFIX)-objdump

	CFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
	CFLAGS += -DRISCV -DTEST_NUM=1
	CFLAGS += -fno-common -fno-inline
	CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
	CFLAGS += -I${VEXII_PATH}/ -I${VEXII_PATH}/driver

	LDFLAGS += -march=$(MARCH) -mabi=$(MABI) -mcmodel=medany
	LDFLAGS += -nostdlib -nostartfiles -ffreestanding -Wl,-Bstatic,-T,$(VEXII_LD),-Map,$(BUILD)/$(MAIN).map,--print-memory-usage
	LDFLAGS += -L./ -nolibc -lm -lc -lgcc

	RISCV_SOURCE = $(wildcard $(VEXII_PATH)/*.c) $(wildcard $(VEXII_PATH)/*.S)
	ifneq ($(filter simd, $(OPT)),)
		MICO_SOURCES += $(wildcard $(MICO_DIR)/src/mico_simd/*.c)
		RISCV_SOURCE += $(wildcard ${VEXII_PATH}/mico/*.S)
	endif
endif

OBJS := $(MICO_SOURCES) $(RISCV_SOURCE)
OBJS := $(OBJS:.c=.o)
OBJS := $(OBJS:.S=.o)
OBJS := $(OBJS:.s=.o)
OBJS := $(addprefix $(BUILD)/,$(OBJS))

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo "Compiling Source File ($<)..."
	@$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	@echo "Compiling Source File ($<)..."
	@$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))

$(MAIN).elf: $(OBJS) $(MAIN).c
	@echo "Linking Executable (main)..."
	@$(CC) $(CFLAGS) -o $(MAIN).elf $(OBJS) $(MAIN).c $(addprefix -I,$(INCLUDES)) $(LDFLAGS)

$(MAIN).asm : $(MAIN).elf
	@echo "Generating Assembly Code..."
	@$(OBJDUMP) -d $(MAIN).elf > $(MAIN).asm

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
	rm -f *.elf *.asm

compile: $(BUILD) $(MAIN).elf $(MAIN).asm

all: compile