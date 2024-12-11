
RISCV_PREFIX=riscv64-unknown-elf-

MICO_DIR = MiCo-Lib

CC = gcc
# CC = clang

MICO_INCLUDES = $(MICO_DIR)/include
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
	$(CC) $(CFLAGS) -c -o $@ $< $(addprefix -I,$(INCLUDES))

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

build-x86: $(OBJS) $(MAIN).c
	$(CC) $(CFLAGS) -o $(MAIN) $(OBJS) $(MAIN).c $(addprefix -I,$(INCLUDES)) $(LDFLAGS)

build-rv:

run-x86: