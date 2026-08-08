# Headless cycle bench for the Elya Genesis engine.
#
# src/ SYMLINKS the real src/elya_gpt.c — one engine edit is measured and
# shipped from one file. Swap res/model.bin to A/B a blob format against a
# byte-identical engine binary.
#
# The resource is force-rebuilt on every `make build`. SGDK's generated
# dependency files are NOT included by makefile.gen (`#-include $(DEPS)`),
# so a changed res/model.bin does NOT invalidate out/res/bench.o on its
# own. That silently produced a ROM with the OLD blob and a NEW engine here
# once already, and the measurement it gave looked like a clean null result.
# Rescomp on a 1.1 MB BIN costs under a second; correctness is worth it.
export GDK := /home/scott/marsdev/mars/m68k-elf

.PHONY: build clean model-id

build: model-id
	@rm -f res/bench.h res/bench.rs out/res/bench.o out/rom.bin
	$(MAKE) -f $(GDK)/makefile.gen

# Printed on every build so the log always says WHICH blob was measured.
model-id:
	@printf 'bench model: %s\n' "$$(md5sum res/model.bin)"

clean:
	$(MAKE) -f $(GDK)/makefile.gen clean
