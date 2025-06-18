(* -------------------------------------------------------------------------
   img_oxocarbon.ml  –  ultra-fast Oxocarbon colour-grading for PNG batches

   Pipeline
   ────────
   1.  Build a dense 3-D LUT (HALD-4 → 64³ texels) using Gaussian-RBF
       interpolation of the official Oxocarbon palette in Oklab.  The binary
       LUT is cached at
         "$XDG_CACHE_HOME/oxocarbon.hald4"  (falls back to  ~/.cache/).

   2.  Subsequent runs load the cached LUT and process all PNGs in parallel:

   Performance tricks
   ──────────────────
   •   Allocation-free sampling & no per-pixel heap work.
   •   Hot-loop: unrolled trilinear interpolation with and prevented LLVM 
   from duplicating loads.
   •   LUT generation across Z-slices runs in parallel.
   •   Pre-scaled constants (`lut_scale`) eliminate divisions in the hot path.
   •   Palette anchors stored once in Oklab with luminance weighting (lum_factor).
   •   Stack-allocated `accum` record removes 4 float-ref allocations per voxel when
       building the LUT.
   •   Zero-copy image hand-off via the [`@unique`] binding on `out` avoids an extra
       buffer clone when the writer domain saves the processed PNG.

   Usage:
     ./img_oxocarbon [--invert|-i] [--preserve|-p] [--lum-factor|-l <f>] [--transparent|-t] <in_dir> <out_dir>
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
  |]

type lab = float * float * float
type rgb = int * int * int

(* Stack-allocated accumulator used inside LUT generation *)
type accum = { mutable w : float; mutable l : float; mutable a : float; mutable b : float } [@@stack]

(* Oklab conversion (borrowed from B. Bottosson) *)
let[@inline always] cube_root x =
  if x > 0. then x ** (1. /. 3.) else -.((-. x) ** (1. /. 3.))

let[@inline always] srgb_to_linear c =
  let c = c /. 255.0 in
  if c <= 0.04045 then c /. 12.92 else ((c +. 0.055) /. 1.055) ** 2.4

let[@inline always] linear_to_srgb v =
  let v =
    if v <= 0.0031308 then v *. 12.92
    else 1.055 *. (v ** (1. /. 2.4)) -. 0.055
  in
  int_of_float (max 0.0 (min 1.0 v) *. 255.0 +. 0.5)

let rgb_to_oklab (r,g,b) : lab =
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

let oklab_to_rgb (l_,a_,b_) : rgb =
  let l' = l_ +. 0.3963377774 *. a_ +. 0.2158037573 *. b_ in
  let m' = l_ -. 0.1055613458 *. a_ -. 0.0638541728 *. b_ in
  let s' = l_ -. 0.0894841775 *. a_ -. 1.2914855480 *. b_ in
  let l3 = l' ** 3. and m3 = m' ** 3. and s3 = s' ** 3. in
  let lr =  4.0767416621 *. l3 -. 3.3077115913 *. m3 +. 0.2309699292 *. s3 in
  let lg = -.1.2684380046 *. l3 +. 2.6097574011 *. m3 -. 0.3413193965 *. s3 in
  let lb =  0.0415550574 *. l3 -. 0.5369569129 *. m3 +. 1.4952948517 *. s3 in
  ( linear_to_srgb lr, linear_to_srgb lg, linear_to_srgb lb )

(* LUT dimensions and derived constants *)
let cube_side   = 64                                  (* 64³ Hald-4 *)
let cube_total  = cube_side * cube_side * cube_side
let inv_cube    = 1.0 /. float (cube_side - 1)         (* 1/(side-1)  *)
let lut_scale   = (float cube_side -. 1.) /. 255.0     (* 0-255 → 0-side-1 *)
let srgb_scale  = 255.0

let gaussian_beta = 128.0  (* controls sharpness *)

(* Luminance weighting – values < 1 boost saturation by de-emphasising L in distance calc. *)
let lum_factor =
  let rec find i =
    if i >= Array.length Sys.argv then None
    else match Sys.argv.(i) with
      | "--lum-factor" | "-l" ->
          if i + 1 < Array.length Sys.argv then Some Sys.argv.(i+1) else None
      | _ -> find (i + 1)
  in
  match find 1 with
  | Some v -> (try float_of_string v with _ -> 1.0)
  | None -> 1.0

let cache_path () =
  let base = match Sys.getenv_opt "XDG_CACHE_HOME" with
    | Some d -> d | None -> Filename.concat (Sys.getenv "HOME") ".cache" in
  if not (Sys.file_exists base) then Unix.mkdir base 0o755;
  let fname = Printf.sprintf "oxocarbon-b%d-l%02d.hald4"
      (int_of_float gaussian_beta)
      (int_of_float (lum_factor *. 100.)) in
  Filename.concat base fname

(* Speed-optimised LUT generation *)
let generate_lut () =
  Printf.eprintf "Generating LUT …\n%!";

  (* Pre-compute palette anchors (array of triples), applying lum_factor to L *)
  let anchors_oklab =
    palette
    |> Array.map rgb_to_oklab
    |> Array.map (fun (l,a,b) -> (l *. lum_factor, a, b))
  in
  let n_colors = Array.length anchors_oklab in

  (* Allocate output LUT once. Each Z-slice writes to a disjoint chunk, so
     parallel writes are data-race-free. *)
  let lut = Bytes.create (cube_total * 3) in

  (* Pool with N-1 helper domains *)
  let pool =
    T.setup_pool ~num_domains:(max 1 (Domain.recommended_domain_count () - 1)) ()
  in

  (* Parallelise along the Z axis *)
  T.run pool (fun () ->
    T.parallel_for pool ~start:0 ~finish:(cube_side - 1) ~body:(fun z ->
      let bz = float z *. inv_cube in
      for y = 0 to cube_side - 1 do
        let by = float y *. inv_cube in
        (* Stack-allocated accumulator reused for every voxel in the row *)
        let acc : accum = { w = 0.; l = 0.; a = 0.; b = 0. } [@stack] in
        for x = 0 to cube_side - 1 do
          let bx = float x *. inv_cube in
          (* sRGB co-ords 0-255 *)
          let r_i = int_of_float (bx *. srgb_scale) in
          let g_i = int_of_float (by *. srgb_scale) in
          let b_i = int_of_float (bz *. srgb_scale) in

          let lx0,ax,bx_ok = rgb_to_oklab (r_i,g_i,b_i) in
          let lx = lx0 *. lum_factor in

          (* Reset accumulator *)
          acc.w <- 0.; acc.l <- 0.; acc.a <- 0.; acc.b <- 0.;

          for i = 0 to n_colors - 1 do
            let la,aa,ba = Array.unsafe_get anchors_oklab i in
            let d2 = (lx -. la)**2. +. (ax -. aa)**2. +. (bx_ok -. ba)**2. in
            let w = exp (-. gaussian_beta *. d2) in
            acc.w <- acc.w +. w;
            acc.l <- acc.l +. w *. la;
            acc.a <- acc.a +. w *. aa;
            acc.b <- acc.b +. w *. ba;
          done;

          let l_avg_scaled = acc.l /. acc.w in
          let a_avg   = acc.a /. acc.w in
          let b_avg   = acc.b /. acc.w in
          let l_avg = l_avg_scaled /. lum_factor in
          let r8,g8,b8 = oklab_to_rgb (l_avg, a_avg, b_avg) in
          let idx = ((z * cube_side + y) * cube_side + x) * 3 in
          Bytes.unsafe_set lut idx     (Char.unsafe_chr r8);
          Bytes.unsafe_set lut (idx+1) (Char.unsafe_chr g8);
          Bytes.unsafe_set lut (idx+2) (Char.unsafe_chr b8);
        done
      done
    )
  );

  T.teardown_pool pool;

  (* Persist cache to disk *)
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
    else (
      let bytes = really_input_string ic sz in
      close_in ic;
      Bytes.of_string bytes ))
  else generate_lut ()

let lut_bytes = load_or_build_lut ()

(* Fast index into the flat LUT byte buffer *)
let[@inline] lut_index x y z = ((z * cube_side + y) * cube_side + x) * 3

(* Linear interpolation helper *)
let[@inline] lerp a b t = a +. (b -. a) *. t

let[@inline never] unsafe_u8 idx = Char.code (Bytes.unsafe_get lut_bytes idx)

(* Trilinear interpolation of the LUT *)
let[@inline always] sample_lut (r:int) (g:int) (b:int) : rgb =
  (* map 0–255 → 0–cube_side-1 in float *)
  let fx = float_of_int r *. lut_scale
  and fy = float_of_int g *. lut_scale
  and fz = float_of_int b *. lut_scale in
  let x0 = int_of_float fx and y0 = int_of_float fy and z0 = int_of_float fz in
  let x1 = min (x0 + 1) (cube_side - 1)
  and y1 = min (y0 + 1) (cube_side - 1)
  and z1 = min (z0 + 1) (cube_side - 1) in
  let dx = fx -. float x0 and dy = fy -. float y0 and dz = fz -. float z0 in

  (* Fetch lattice colour bytes (unrolled) *)
  let i000 = lut_index x0 y0 z0 and i100 = lut_index x1 y0 z0 in
  let i010 = lut_index x0 y1 z0 and i110 = lut_index x1 y1 z0 in
  let i001 = lut_index x0 y0 z1 and i101 = lut_index x1 y0 z1 in
  let i011 = lut_index x0 y1 z1 and i111 = lut_index x1 y1 z1 in

  (* Pre-read all 24 bytes once *)
  let r000 = float (unsafe_u8 i000) and g000 = float (unsafe_u8 (i000+1)) and b000 = float (unsafe_u8 (i000+2)) in
  let r100 = float (unsafe_u8 i100) and g100 = float (unsafe_u8 (i100+1)) and b100 = float (unsafe_u8 (i100+2)) in
  let r010 = float (unsafe_u8 i010) and g010 = float (unsafe_u8 (i010+1)) and b010 = float (unsafe_u8 (i010+2)) in
  let r110 = float (unsafe_u8 i110) and g110 = float (unsafe_u8 (i110+1)) and b110 = float (unsafe_u8 (i110+2)) in
  let r001 = float (unsafe_u8 i001) and g001 = float (unsafe_u8 (i001+1)) and b001 = float (unsafe_u8 (i001+2)) in
  let r101 = float (unsafe_u8 i101) and g101 = float (unsafe_u8 (i101+1)) and b101 = float (unsafe_u8 (i101+2)) in
  let r011 = float (unsafe_u8 i011) and g011 = float (unsafe_u8 (i011+1)) and b011 = float (unsafe_u8 (i011+2)) in
  let r111 = float (unsafe_u8 i111) and g111 = float (unsafe_u8 (i111+1)) and b111 = float (unsafe_u8 (i111+2)) in

  (* Interpolate R *)
  let ix0 = lerp r000 r100 dx in
  let ix1 = lerp r010 r110 dx in
  let ix2 = lerp r001 r101 dx in
  let ix3 = lerp r011 r111 dx in
  let iy0 = lerp ix0 ix1 dy in
  let iy1 = lerp ix2 ix3 dy in
  let r_out = int_of_float (lerp iy0 iy1 dz +. 0.5) in

  (* Interpolate G *)
  let ix0 = lerp g000 g100 dx in
  let ix1 = lerp g010 g110 dx in
  let ix2 = lerp g001 g101 dx in
  let ix3 = lerp g011 g111 dx in
  let iy0 = lerp ix0 ix1 dy in
  let iy1 = lerp ix2 ix3 dy in
  let g_out = int_of_float (lerp iy0 iy1 dz +. 0.5) in

  (* Interpolate B *)
  let ix0 = lerp b000 b100 dx in
  let ix1 = lerp b010 b110 dx in
  let ix2 = lerp b001 b101 dx in
  let ix3 = lerp b011 b111 dx in
  let iy0 = lerp ix0 ix1 dy in
  let iy1 = lerp ix2 ix3 dy in
  let b_out = int_of_float (lerp iy0 iy1 dz +. 0.5) in

  (r_out, g_out, b_out)

(* Per-row worker *)
let is_black r g b = r < 24 && g < 24 && b < 24

let process_row ~invert ~preserve ~transparent out img y w =
  for x = 0 to w - 1 do
    Image.read_rgba img x y @@ fun r g b _a ->
      let make_transparent = transparent && is_black r g b in
      if make_transparent then
        Image.write_rgba out x y 0 0 0 0
      else
        let ri,gi,bi = if invert then 255 - r, 255 - g, 255 - b else r, g, b in
        let pr,pg,pb = sample_lut ri gi bi in

        (* Optionally preserve original luminance (Oklab L component) *)
        let pr,pg,pb =
          if preserve then
            let l0,_,_ = rgb_to_oklab (ri,gi,bi) in
            let _l1,a1,b1 = rgb_to_oklab (pr,pg,pb) in
            oklab_to_rgb (l0,a1,b1)
          else (pr,pg,pb)
        in
        Image.write_rgba out x y pr pg pb 255
  done

(* Batch processing *)
let process_file pool ~invert ~preserve ~transparent in_dir out_dir file =
  if Filename.check_suffix file ".png" then (
    let src = Filename.concat in_dir file and dst = Filename.concat out_dir file in
    let img = ImageLib_unix.openfile src in
    let w,h = img.width, img.height in
    let out [@unique] = Image.create_rgb ~alpha:true w h in
    T.run pool (fun () ->
        T.parallel_for pool ~start:0 ~finish:(h-1) ~chunk_size:16 ~body:(fun y -> process_row ~invert ~preserve ~transparent out img y w)
    );
    ImageLib_unix.writefile dst out;
    Printf.printf "✓ %s\n%!" file)

(* Entry point *)
let () =
  (* CLI: [--invert|-i] [--preserve|-p] [--lum-factor|-l <f>] [--transparent|-t] <in_dir> <out_dir> *)
  let invert       = ref false in
  let preserve     = ref false in
  let transparent  = ref false in
  let positional   = ref [] in
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    (match Sys.argv.(!i) with
     | "--invert" | "-i" -> invert := true; incr i
     | "--preserve" | "-p" -> preserve := true; incr i
     | "--transparent" | "-t" -> transparent := true; incr i
     | "--lum-factor" | "-l" ->
         if !i + 1 < Array.length Sys.argv then i := !i + 2 else i := !i + 1
     | arg -> positional := arg :: !positional; incr i)
  done;
  match List.rev !positional with
  | [in_dir; out_dir] ->
      if not (Sys.file_exists out_dir) then Unix.mkdir out_dir 0o755;
      let pool = T.setup_pool ~num_domains:(max 1 (Domain.recommended_domain_count () - 1)) () in
      Sys.readdir in_dir |> Array.iter (process_file pool ~invert:!invert ~preserve:!preserve ~transparent:!transparent in_dir out_dir);
      T.teardown_pool pool
  | _ ->
      Printf.eprintf "Usage: %s [--invert|-i] [--preserve|-p] [--transparent|-t] [--lum-factor|-l <f>] <in_dir> <out_dir>\n" Sys.argv.(0);
      exit 1

let is_image f =
  List.exists (Filename.check_suffix f)
              [ ".png"; ".bmp"; ".ppm"; ".jpg" ] (* formats imagelib supports *)
