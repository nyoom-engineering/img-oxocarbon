# Makefile ─ build + run Oxocarbon PNG batcher
# dirs:  caida-original/   (source PNGs)
#        out/              (results go here)

OCAMLFIND   = ocamlfind
OCAMLOPTFLG = -O3 -unsafe -unbox-closures -unboxed-types
PKGS        = imagelib.unix,domainslib
SRC         = img_oxocarbon.ml
BIN         = img_oxocarbon

.PHONY: all run test clean

all: $(BIN) run test

$(BIN): $(SRC)
	$(OCAMLFIND) ocamlopt $(OCAMLOPTFLG) -package $(PKGS) -linkpkg \
		-o $@ $<

run: | out
	./$(BIN) --invert --preserve --transparent caida-original out

test: | out
	./$(BIN) --preserve --lum-factor 0.7 test-img out

out:
	mkdir -p $@

clean:
	rm -f $(BIN)
	rm -rf out
