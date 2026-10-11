// The systolic MXU's arithmetic cells (otpu_mxu IMPL 2), technology-independent behavioural
// models: the simulators' (Verilator), the ASIC flows' and any part without the FPGA families'
// DSP cells. rtl/tech/xilinx7 (DSP48E1) and rtl/tech/xilinxus (DSP48E2) implement the same
// modules, cycle for cycle; the file list picks one (opentpu/target.py, docs/targets.md).

// One position of the systolic MXU (IMPL 2): the activation registered (as a DSP's B register),
// the two weight streams lo / hi, one picked by the registered select sel, the product registered
// (M) and the chain's running sum P = PCIN + M (FIRST: P = M), registered unless LAST (the column
// end registers a chain's last sum). WREG registers the weights too (wlo, whi are their next
// values); sel is the select's next value. The pipeline is the DSP48's (rtl/tech/xilinx7 says
// why each register is where it is).
module otpu_pe #(parameter bit FIRST = 1'b0, parameter bit LAST = 1'b0, parameter bit WREG = 1'b0) (
  input  logic               clk,
  input  logic               en,
  input  logic [7:0]         act,
  input  logic [7:0]         wlo,               // WREG: the weights from the next cycle on
  input  logic [7:0]         whi,
  input  logic               sel,               // the select from the next cycle on
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic signed [23:0] p
);
  logic [7:0] br, wlr, whr;
  logic       selr;
  logic signed [15:0] m;
  wire  signed [23:0] pn = (FIRST ? 24'sd0 : $signed(pcin[23:0])) + 24'(m);
  wire  [7:0] wl = WREG ? wlr : wlo, wh = WREG ? whr : whi;
  always_ff @(posedge clk) if (en) begin
    br <= act;
    selr <= sel;
    wlr <= wlo;
    whr <= whi;
    m <= 16'(int'($signed(br)) * int'($signed(selr ? wh : wl)));
  end
  if (LAST) begin : g_comb
    assign p = pn;
  end else begin : g_preg
    always_ff @(posedge clk) if (en) p <= pn;
  end
  assign pcout = LAST ? 48'd0 : 48'(p);              // (a chain's last cascades nowhere)
endmodule


// The systolic MXU's column end (IMPL 2): per sub-block b the chains' ends a[b] and d[b] (the
// LAST positions' unregistered sums) are registered with m_b (mb[4b+3:4b], unsigned), summed and
// multiplied into v[b]; one cycle later s = v0 + v1 + v2 + v3 (its low SW bits). Exact: every
// value fits 48 bits. rtl/tech/xilinx7 maps it onto four cascaded DSP48E1s.
module otpu_colend #(parameter int SW = 23) (
  input  logic                 clk,
  input  logic                 en,
  input  logic [3:0][23:0]     a,        // sub-block b's chain ends: the pre-adder's A ...
  input  logic [3:0][23:0]     d,        // ... and D (signed)
  input  logic [15:0]          mb,       // m_3 .. m_0, with a / d
  output logic signed [SW-1:0] s
);
  // the DSP48E1's pre-adder is 25 bits wide (rtl/tech/xilinx7): every technology takes the same
  // parameters
  initial if (SW > 25) $fatal(1, "otpu_colend: SW %0d > 25", SW);
  logic [3:0][23:0] ar, dr;
  logic [15:0] mr;
  logic signed [47:0] v [4];
  always_ff @(posedge clk) if (en) begin
    ar <= a; dr <= d; mr <= mb;
    for (int b = 0; b < 4; b++)
      v[b] <= (48'($signed(ar[b])) + 48'($signed(dr[b]))) * 48'(mr[4*b +: 4]);
    s <= SW'(v[0] + v[1] + v[2] + v[3]);
  end
endmodule
