# Headless cycle bench for the Elya Genesis engine.
# src/ SYMLINKS the real src/elya_gpt.c — one engine edit, measured and
# shipped from one file. Swap res/model.bin to A/B a blob format.
export GDK := /home/scott/marsdev/mars/m68k-elf

.PHONY: build clean
build:
	$(MAKE) -f $(GDK)/makefile.gen
clean:
	$(MAKE) -f $(GDK)/makefile.gen clean
