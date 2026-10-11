// The technology RAMs on SRAM macros (an ASIC's rtl/tech; the generic file infers the same
// modules). Each RAM is tiled onto leaf macros of at most WMAX bits by NMAX words: width tiles side
// by side, depth tiles (N a multiple of NMAX) with a registered read bank select. The leaves
// (otpu_sram_1r1w, otpu_sram_1rw) are generated per design by tools/asic/flow.py, one branch per
// macro size the design uses (asic/<pdk>'s macros: tools/asic/fakeram.py for NanGate45).
//
// The macros' semantics are the generic RAMs': registered reads with an enable (the output holds
// otherwise), writes masked per bit. A 1R1W macro's read of the word written at the same edge is
// undefined; no user depends on it (otpu_tmem bypasses it). STYLE and ZERO are the FPGA's and the
// simulation's, unused here.
package otpu_tech_asic;
  localparam int WMAX = 128;    // bits per macro
  localparam int NMAX = 4096;   // words per macro
endpackage

// one tiled RAM: KIND 0 1R1W, 1 1RW (a: ra; en: re | we)
module otpu_ram_tiled #(
  parameter int W    = 32,
  parameter int N    = 1024,
  parameter int KIND = 0
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [W-1:0]         wm,      // per bit
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  import otpu_tech_asic::*;
  localparam int AW = $clog2(N);
  localparam int NW = (W + WMAX - 1) / WMAX;          // width tiles
  localparam int TN = (N < NMAX) ? N : NMAX;          // words per tile
  localparam int NN = N / TN;                         // depth tiles
  localparam int TA = $clog2(TN);
  initial if (NN * TN != N) $fatal(1, "otpu_ram_tiled: %0d words is not a multiple of %0d", N, NMAX);
  // each depth tile's read data (unpacked: Verilator 5.047 read q[rbank] of a packed [NN][W]
  // wrongly here, W 40, N 8192; tests/test_tech_ram.py)
  logic [W-1:0] q [NN];
  logic [(NN > 1 ? $clog2(NN) : 1)-1:0] rbank;   // the tile the last read read
  // the depth tile an address is in (0 with one tile)
  localparam int BW = (NN > 1) ? $clog2(NN) : 1;
  logic [BW-1:0] rt, wt;
  if (NN > 1) begin : g_sel
    assign rt = ra[AW-1:TA];
    assign wt = wa[AW-1:TA];
  end else begin : g_nosel
    assign rt = '0;
    assign wt = '0;
  end
  for (genvar d = 0; d < NN; d++) begin : g_d
    wire rsel = (rt == BW'(d));
    wire wsel = (wt == BW'(d));
    for (genvar c = 0; c < NW; c++) begin : g_c
      localparam int CW = (c == NW - 1) ? W - c * WMAX : WMAX;
      if (KIND == 0) begin : g_2p
        otpu_sram_1r1w #(.W(CW), .N(TN)) u_m (
          .clk, .re(re && rsel), .ra(ra[TA-1:0]), .we(we && wsel), .wa(wa[TA-1:0]),
          .wd(wd[c*WMAX +: CW]), .wm(wm[c*WMAX +: CW]), .rd(q[d][c*WMAX +: CW]));
      end else begin : g_1p
        // one port: the write's address when writing, the read's otherwise
        otpu_sram_1rw #(.W(CW), .N(TN)) u_m (
          .clk, .en((re && rsel) || (we && wsel)), .we(we && wsel),
          .a(we ? wa[TA-1:0] : ra[TA-1:0]), .wd(wd[c*WMAX +: CW]), .wm(wm[c*WMAX +: CW]),
          .rd(q[d][c*WMAX +: CW]));
      end
    end
  end
  if (NN > 1) begin : g_bank
    always_ff @(posedge clk) if (re) rbank <= rt;
    assign rd = q[rbank];
  end else begin : g_one
    assign rbank = '0;
    assign rd = q[0];
  end
endmodule

module otpu_ram_1r1w #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  otpu_ram_tiled #(.W(W), .N(N), .KIND(0)) u_t (
    .clk, .we, .wm('1), .wa, .wd, .re, .ra, .rd);
endmodule

module otpu_ram_1r1w_be #(
  parameter int    W     = 64,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [W/8-1:0]       wbe,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  logic [W-1:0] wm;
  always_comb for (int k = 0; k < W / 8; k++) wm[8*k +: 8] = {8{wbe[k]}};
  otpu_ram_tiled #(.W(W), .N(N), .KIND(0)) u_t (
    .clk, .we, .wm, .wa, .wd, .re, .ra, .rd);
endmodule

module otpu_ram_1rw #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter string STYLE = "block",
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 en,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] a,
  input  logic [W-1:0]         wd,
  output logic [W-1:0]         rd
);
  otpu_ram_tiled #(.W(W), .N(N), .KIND(1)) u_t (
    .clk, .we(en && we), .wm('1), .wa(a), .wd, .re(en && !we), .ra(a), .rd);
endmodule
