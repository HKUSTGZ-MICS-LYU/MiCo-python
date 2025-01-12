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
RISCV_PREFIX = riscv64-unknown-elf
RISCV_SOURCE ?= 

# Targets
# VexiiRiscv Related
include $(MICO_DIR)/targets/vexii.mk
# Chipyard Related
include $(MICO_DIR)/targets/rocket.mk

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

run_host: $(MAIN).elf
	./$(MAIN).elf

all: compile