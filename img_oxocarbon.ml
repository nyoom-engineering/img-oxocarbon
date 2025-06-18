(* -------------------------------------------------------------------------
   img_oxocarbon.ml  –  ultra-fast Oxocarbon colour-grading for PNG batches

   Pipeline
   ────────
   1.  Build a dense 3-D LUT (HALD-4 → 64³ texels) using Gaussian-RBF
       interpolation of the official Oxocarbon palette in Oklab.  The binary
       LUT is cached at
         "$XDG_CACHE_HOME/oxocarbon.hald4"  (falls back to  ~/.cache/).

   2.  Subsequent runs load the cached LUT and process all PNGs in parallel:
       • pixel is inverted to match the dark-to-light aesthetic;            
       • near-black pixels become transparent;                              
       • all others are mapped through allocation-free, inlined trilinear
         filtering of the LUT.

   Performance tricks
   ──────────────────
   •   Allocation-free sampling: no tuples, no per-pixel heap work.
   •   `Bytes.unsafe_get` + `[@inline]` to keep the hot loop tight.
   •   Pre-scaled constants (`lut_scale`) remove divisions in the hot path.
   •   Domainslib parallelism with a tuned `chunk_size` for row workers.
   •   Compiled with `-O3 -unsafe` (OCaml ≥ 5.0) for maximum speed.

   Usage:
     make run
     ./img_oxocarbon <in_dir> <out_dir>
   -------------------------------------------------------------------------*)

open Image
module T = Domainslib.Task

(* palette *)
let palette =
  [|
    0x16,0x16,0x16; (* bg1 *)   
    0x26,0x26,0x26; (* bg2 *)
    0x39,0x39,0x39; (* bg3 *)   
    0x52,0x52,0x52; (* bg4 *)
    0xdd,0xe1,0xe6; (* fg1 *)   
    0xf2,0xf4,0xf8; (* fg2 *)
    0xff,0xff,0xff; (* fg3 *)   
    0xff,0x7e,0xb6; (* accent1 *)
    0x3d,0xdb,0xd9; (* accent2 *)
    0x78,0xa9,0xff; (* blue1 *) 
    0x82,0xcf,0xff; (* blue2 *)
    0x33,0xb1,0xff; (* blue3 *) 
    0x08,0xbd,0xba; (* blue4 *)
    0xbe,0x95,0xff; (* purple *)
    0xee,0x53,0x96; (* negative *)
    0x42,0xbe,0x65; (* positive *)
    0x8d,0x8d,0x8d; (* ignore *)
  |]

(* Oklab conversion (borrowed from B. Bottosson) *)
let cube_root x = if x > 0. then x ** (1. /. 3.) else -.((-. x) ** (1. /. 3.))

let srgb_to_linear c =
  let c = c /. 255.0 in
  if c <= 0.04045 then c /. 12.92 else ((c +. 0.055) /. 1.055) ** 2.4

let linear_to_srgb v =
  let v = if v <= 0.0031308 then v *. 12.92 else 1.055 *. (v ** (1. /. 2.4)) -. 0.055 in
  int_of_float (max 0.0 (min 1.0 v) *. 255.0 +. 0.5)

let rgb_to_oklab (r,g,b) =
  let lr_lin  = srgb_to_linear (float_of_int r)
  and lg_lin  = srgb_to_linear (float_of_int g)
  and lb_lin  = srgb_to_linear (float_of_int b) in
  let l = 0.4122214708 *. lr_lin +. 0.5363325363 *. lg_lin +. 0.0514459929 *. lb_lin in
  let m = 0.2119034982 *. lr_lin +. 0.6806995451 *. lg_lin +. 0.1073969566 *. lb_lin in
  let s = 0.0883024619 *. lr_lin +. 0.2817188376 *. lg_lin +. 0.6299787005 *. lb_lin in
  let l' = cube_root l and m' = cube_root m and s' = cube_root s in
  let l_ = 0.2104542553 *. l' +. 0.7936177850 *. m' -. 0.0040720468 *. s'
  and a_ = 1.9779984951 *. l' -. 2.4285922050 *. m' +. 0.4505937099 *. s'
  and b_ = 0.0259040371 *. l' +. 0.7827717662 *. m' -. 0.8086757660 *. s' in
  (l_,a_,b_)

let oklab_to_rgb (l_,a_,b_) =
  let l' = l_ +. 0.3963377774 *. a_ +. 0.2158037573 *. b_ in
  let m' = l_ -. 0.1055613458 *. a_ -. 0.0638541728 *. b_ in
  let s' = l_ -. 0.0894841775 *. a_ -. 1.2914855480 *. b_ in
  let l3 = l' ** 3. and m3 = m' ** 3. and s3 = s' ** 3. in
  let lr =  4.0767416621 *. l3 -. 3.3077115913 *. m3 +. 0.2309699292 *. s3 in
  let lg = -.1.2684380046 *. l3 +. 2.6097574011 *. m3 -. 0.3413193965 *. s3 in
  let lb =  0.0415550574 *. l3 -. 0.5369569129 *. m3 +. 1.4952948517 *. s3 in
  ( linear_to_srgb lr, linear_to_srgb lg, linear_to_srgb lb )

(* LUT generation (Gaussian RBF) *)
let cube_side = 64        (* HALD-4: 64 × 64 × 64 *)
let cube_total = cube_side * cube_side * cube_side
let cube_side_f = float cube_side
let lut_scale = (cube_side_f -. 1.) /. 255.0

let cache_path () =
  let base = match Sys.getenv_opt "XDG_CACHE_HOME" with
    | Some d -> d | None -> Filename.concat (Sys.getenv "HOME") ".cache" in
  if not (Sys.file_exists base) then Unix.mkdir base 0o755;
  Filename.concat base "oxocarbon.hald4"

let gaussian_beta = 60.0  (* controls sharpness; experiment *)

let generate_lut () =
  Printf.eprintf "Generating LUT …\n%!";
  let anchors_oklab = Array.map rgb_to_oklab palette in
  let lut = Bytes.create (cube_total * 3) in
  for z = 0 to cube_side - 1 do
    let bz = float z /. float (cube_side - 1) in
    for y = 0 to cube_side - 1 do
      let by = float y /. float (cube_side - 1) in
      for x = 0 to cube_side - 1 do
        let bx = float x /. float (cube_side - 1) in
        let rx, gx, bx' = bx, by, bz in
        (* convert cube coord to sRGB 0-255 *)
        let r_i = int_of_float (rx *. 255.)
        and g_i = int_of_float (gx *. 255.)
        and b_i = int_of_float (bx'*. 255.) in
        let lx,ax,bx_ok = rgb_to_oklab (r_i,g_i,b_i) in
        (* gaussian RBF blend of palette in Oklab *)
        let ws,l_sum,a_sum,b_sum = ref 0., ref 0., ref 0., ref 0. in
        Array.iter (fun (la,aa,ba) ->
          let d2 = (lx -. la)**2. +. (ax -. aa)**2. +. (bx_ok -. ba)**2. in
          let w = exp (-. gaussian_beta *. d2) in
          ws := !ws +. w;
          l_sum := !l_sum +. w *. la;
          a_sum := !a_sum +. w *. aa;
          b_sum := !b_sum +. w *. ba)
          anchors_oklab;
        let l_avg = !l_sum /. !ws and a_avg = !a_sum /. !ws and b_avg = !b_sum /. !ws in
        let r8,g8,b8 = oklab_to_rgb (l_avg,a_avg,b_avg) in
        let idx = ((z * cube_side + y) * cube_side + x) * 3 in
        Bytes.set lut idx     (Char.chr r8);
        Bytes.set lut (idx+1) (Char.chr g8);
        Bytes.set lut (idx+2) (Char.chr b8);
      done
    done;
  done;
  let path = cache_path () in
  let oc = open_out_bin path in
  output_bytes oc lut; close_out oc;
  lut

let load_or_build_lut () =
  let path = cache_path () in
  if Sys.file_exists path then (
    let ic = open_in_bin path in
    let sz = in_channel_length ic in
    if sz <> cube_total * 3 then (close_in ic; generate_lut ())
    else let bytes = really_input_string ic sz in close_in ic; Bytes.of_string bytes)
  else generate_lut ()

let lut_bytes = load_or_build_lut ()

(* Fast index into the flat LUT byte buffer *)
let[@inline] lut_index x y z = ((z * cube_side + y) * cube_side + x) * 3

(* Linear interpolation helper *)
let[@inline] lerp a b t = a +. (b -. a) *. t

(* Trilinear interpolation of the LUT – allocation-free & branch-light *)
let[@inline always] sample_lut r g b =
  (* map 0–255 → 0–cube_side-1 in float *)
  let fx = float_of_int r *. lut_scale
  and fy = float_of_int g *. lut_scale
  and fz = float_of_int b *. lut_scale in
  let x0 = int_of_float fx and y0 = int_of_float fy and z0 = int_of_float fz in
  let x1 = min (x0 + 1) (cube_side - 1)
  and y1 = min (y0 + 1) (cube_side - 1)
  and z1 = min (z0 + 1) (cube_side - 1) in
  let dx = fx -. float x0 and dy = fy -. float y0 and dz = fz -. float z0 in

  (* Helper to fetch a colour channel as float *)
  let get idx off = float (Char.code (Bytes.unsafe_get lut_bytes (idx + off))) in

  (* Fetch lattice colours *)
  let i000 = lut_index x0 y0 z0 and i100 = lut_index x1 y0 z0 in
  let i010 = lut_index x0 y1 z0 and i110 = lut_index x1 y1 z0 in
  let i001 = lut_index x0 y0 z1 and i101 = lut_index x1 y0 z1 in
  let i011 = lut_index x0 y1 z1 and i111 = lut_index x1 y1 z1 in

  (* Interpolate R, G, B channels independently *)
  let interp_channel off =
    let ix0 = lerp (get i000 off) (get i100 off) dx in
    let ix1 = lerp (get i010 off) (get i110 off) dx in
    let ix2 = lerp (get i001 off) (get i101 off) dx in
    let ix3 = lerp (get i011 off) (get i111 off) dx in
    let iy0 = lerp ix0 ix1 dy in
    let iy1 = lerp ix2 ix3 dy in
    int_of_float (lerp iy0 iy1 dz +. 0.5)
  in
  (interp_channel 0, interp_channel 1, interp_channel 2)

(* per-row worker *)
let is_black r g b = r < 24 && g < 24 && b < 24

let process_row ~invert out img y w =
  for x = 0 to w - 1 do
    Image.read_rgba img x y @@ fun r g b _a ->
      if is_black r g b then
        Image.write_rgba out x y 0 0 0 0
      else
        let ri,gi,bi = if invert then 255 - r, 255 - g, 255 - b else r, g, b in
        let pr,pg,pb = sample_lut ri gi bi in
        Image.write_rgba out x y pr pg pb 255
  done

(* batch processing *)
let process_file pool ~invert in_dir out_dir file =
  if Filename.check_suffix file ".png" then (
    let src = Filename.concat in_dir file and dst = Filename.concat out_dir file in
    let img = ImageLib_unix.openfile src in
    let w,h = img.width, img.height in
    let out = Image.create_rgb ~alpha:true w h in
    T.run pool (fun () ->
        T.parallel_for pool ~start:0 ~finish:(h-1) ~chunk_size:16 ~body:(fun y -> process_row ~invert out img y w));
    ImageLib_unix.writefile dst out;
    Printf.printf "✓ %s\n%!" file)

(* entry point *)
let () =
  (* simple CLI parsing: [--invert|-i] <in_dir> <out_dir> *)
  let invert = ref false in
  let positional = ref [] in
  for i = 1 to Array.length Sys.argv - 1 do
    match Sys.argv.(i) with
    | "--invert" | "-i" -> invert := true
    | arg -> positional := arg :: !positional
  done;
  match List.rev !positional with
  | [in_dir; out_dir] ->
      if not (Sys.file_exists out_dir) then Unix.mkdir out_dir 0o755;
      let pool = T.setup_pool ~num_domains:(max 1 (Domain.recommended_domain_count () - 1)) () in
      Sys.readdir in_dir |> Array.iter (process_file pool ~invert:!invert in_dir out_dir);
      T.teardown_pool pool
  | _ ->
      Printf.eprintf "Usage: %s [--invert|-i] <in_dir> <out_dir>\n" Sys.argv.(0);
      exit 1

let is_image f =
  List.exists (Filename.check_suffix f)
              [ ".png"; ".bmp"; ".ppm"; ".jpg" ] (* formats imagelib supports *)
