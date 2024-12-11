
RISCV_PREFIX=riscv64-unknown-elf-

MICO_DIR = MiCo-Lib

CC = gcc
# CC = clang

MICO_INCLUDES = $(MICO_DIR)/include $(MICO_DIR)/test
MICO_SOURCES = $(wildcard $(MICO_DIR)/src/*.c)

CFLAGS = -O3
LDFLAGS = -lm

INCLUDES = $(MICO_INCLUDES) ./

BUILD = build
MAIN = main

OBJS := $(MICO_SOURCES)
OBJS := $(OBJS:.c=.o)
OBJS := $(addprefix $(BUILD)/,$(OBJS))

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	@echo "Compiling Source File ($<)..."
	@$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

compile: $(OBJS) $(MAIN).c
	@echo "Linking Executable (main)..."
	@$(CC) $(CFLAGS) -o $(MAIN).elf $(OBJS) $(MAIN).c $(addprefix -I,$(INCLUDES)) $(LDFLAGS)