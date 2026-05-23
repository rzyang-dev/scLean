# scLean Makefile

.PHONY: clean clean-src clean-all

clean:
	rm -f src/*.o src/*.so src/*/*.o src/*/*.so
	rm -rf scLean.Rcheck

clean-src:
	find src -name '*.o' -delete
	find src -name '*.so' -delete

clean-all: clean
	rm -f src/Makevars src/Makevars.win
	rm -rf tmp/
