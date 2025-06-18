# Oxocarbon Image Batch Processor

> Ultra-fast, palette-accurate colour-grading for images – powered by the **Oxocarbon** palette and written in OCaml.

You should probably use lutgen-rs, the output is the same, but this one is fun

## Features

* One-time LUT build – A dense 64³ HALD‐4 lookup-table is generated once using Gaussian-RBF interpolation in Oklab space and cached.
* Blazing-fast processing – Allocation-free, inlined trilinear filtering & parallelism chew through whole folders of PNGs in seconds.
* Zero dependencies at runtime – The tool outputs standard 32-bit PNG files; no external libraries needed once built.

## Installation

### Dependencies

* **OCaml ≥ 5.0**
* [`opam`](https://opam.ocaml.org) with the following libraries:
  * `imagelib` (`imagelib.unix` flavour)
  * `domainslib`

A one-liner to obtain everything on a new switch:

```bash
opam switch create 5.2.0  # or any ≥ 5.0 compiler
opam install imagelib domainslib
```

### Run

```bash
# inside the caida/ directory
make          # compiles img_oxocarbon, runs on caida-original/ and puts results in out/
make run      # builds (if needed) and colour-grades all PNGs in caida-original/ → out/

# or manually
ocamlfind ocamlopt -O3 -thread -unsafe \
  -package imagelib.unix,domainslib -linkpkg \
  -o img_oxocarbon img_oxocarbon.ml

./img_oxocarbon <input_dir> <output_dir>
```

Processing is embarrassingly parallel: by default the program launches `n-1` domains (where *n* = logical cores) and splits work in 16-row chunks.

## License

This project is vendored under the MIT liscense

The Oxocarbon palette belongs to its respective authors; used here under the same terms as the original theme.

## Acknowledgements

* [Björn Ottosson](https://bottosson.github.io/) – *Oklab* perceptual colour space.
* [Ozwaldorf](https://github.com/ozwaldorf/lutgen-rs/tree/main) - Lutgen-RS# img-oxocarbon
