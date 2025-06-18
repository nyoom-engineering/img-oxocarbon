# Makefile ─ build + run Oxocarbon PNG batcher
# dirs:  caida-original/   (source PNGs)
#        out/              (results go here)

OCAMLFIND   = ocamlfind
OCAMLOPTFLG = -O3 -unsafe -unbox-closures -unboxed-types
PKGS        = imagelib.unix,domainslib
SRC         = img_oxocarbon.ml
BIN         = img_oxocarbon

.PHONY: all run clean

all: $(BIN) run

$(BIN): $(SRC)
	$(OCAMLFIND) ocamlopt $(OCAMLOPTFLG) -package $(PKGS) -linkpkg \
		-o $@ $<

run: | out
	./$(BIN) --invert --preserve caida-original out

out:
	mkdir -p $@

clean:
	rm -f $(BIN)
	rm -rf out
