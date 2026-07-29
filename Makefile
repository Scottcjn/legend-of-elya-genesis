# Legend of Elya — Genesis port
# Builds with marsdev (local m68k-elf-gcc + SGDK 1.81), no Docker needed.
#   toolchain: /home/scott/marsdev/mars

export GDK := /home/scott/marsdev/mars/m68k-elf

.PHONY: build run clean

build:
	$(MAKE) -f $(GDK)/makefile.gen

run: build
	flatpak run com.retrodev.blastem $(CURDIR)/out/rom.bin

clean:
	$(MAKE) -f $(GDK)/makefile.gen clean
