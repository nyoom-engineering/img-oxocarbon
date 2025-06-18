# Oxocarbon Image Batch Processor

> Ultra-fast, palette-accurate colour-grading for images – powered by the **Oxocarbon** palette and written in OCaml.

You should probably use lutgen-rs, the output is the same, but this one is fun

<figure>
  <img alt="ries-t2" width="375" src="https://github.com/user-attachments/assets/bd4a0ce6-d578-470a-a5f3-f786bd964796" />
  <figcaption>ries-t2</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="ries-t" width="375" src="https://github.com/user-attachments/assets/17813987-63c7-437b-b5a0-96c7461c7582" />
  <figcaption>ries-t</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="med-gr-l-4" width="375" src="https://github.com/user-attachments/assets/55efa7c4-f91e-4c09-9837-95610e9c13c9" />
  <figcaption>med-gr-l-4</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="lar-gr-l-13" width="375" src="https://github.com/user-attachments/assets/2c79a562-e7a9-426b-87d3-ab45de4b303c" />
  <figcaption>lar-gr-l-13</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="lar-gr-l-7" width="375" src="https://github.com/user-attachments/assets/be9f5343-4318-4f8d-94f2-94ce14400306" />
  <figcaption>lar-gr-l-7</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="lar-gr-l-1" width="375" src="https://github.com/user-attachments/assets/a7880c3a-787f-4e27-9731-8dec5b2d6efb" />
  <figcaption>lar-gr-l-1</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="champ2" width="375" src="https://github.com/user-attachments/assets/9eb404ac-4c1a-4d30-9ce5-105fb75e69fa" />
  <figcaption>champ2</figcaption>
</figure>

<br>
<br>

<figure>
  <img alt="a-root-rtt-05-key" width="375" src="https://github.com/user-attachments/assets/4123009f-4dd5-443f-b1f0-8e0b7d32055f" />
  <figcaption>a-root-rtt-05-key</figcaption>
</figure>

<br>
<br>

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

* [Björn Ottosson](https://bottosson.github.io/) – Author of Oklab perceptual colour space.
* [Ozwaldorf](https://github.com/ozwaldorf/lutgen-rs/tree/main) - Author of lutgen-rs
* [Caida](https://www.caida.org) - Source of sample images